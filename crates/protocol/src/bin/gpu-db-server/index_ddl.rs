// Legacy index DDL ownership. This is not a product execution path.

use super::{
    schema_permission_error, text_column, validate_unique_indexes, write_command_complete,
    write_error, write_single_row, CatalogCommentTarget, CatalogIndex, Command, ErrorField,
    ReadWrite, SchemaPrivilege, Session,
};
use std::collections::BTreeSet;
use std::io;

fn psql_describe_indexes_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_indexes_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_indexes_catalog_query_schema_filter(canonical: &str) -> Option<String> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','s','') and n.nspname operator(pg_catalog.~) '^(";
    let suffix = ")$' collate pg_catalog.default order by 1,2";
    let namespace = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    (namespace == "public").then(|| namespace.to_string())
}

fn psql_describe_index_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .map(|index| {
            (
                index.name.clone(),
                vec![
                    Some("public".to_string()),
                    Some(index.name.clone()),
                    Some("index".to_string()),
                    Some("postgres".to_string()),
                    Some(index.table.clone()),
                ],
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    rows.into_iter().map(|(_, row)| row).collect()
}

fn psql_describe_index_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .map(|index| {
            (
                index.name.clone(),
                vec![
                    Some("public".to_string()),
                    Some(index.name.clone()),
                    Some("index".to_string()),
                    Some("postgres".to_string()),
                    Some(index.table.clone()),
                    Some("permanent".to_string()),
                    Some("btree".to_string()),
                    None,
                    session
                        .comments
                        .get(&CatalogCommentTarget::Index {
                            index: index.name.clone(),
                        })
                        .cloned(),
                ],
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    rows.into_iter().map(|(_, row)| row).collect()
}

pub(super) fn try_execute_index_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    let (columns, rows) = if canonical == psql_describe_indexes_verbose_catalog_query() {
        (
            vec![
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Table"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            psql_describe_index_verbose_rows(session),
        )
    } else if canonical == psql_describe_indexes_catalog_query()
        || psql_describe_indexes_catalog_query_schema_filter(canonical).is_some()
    {
        (
            vec![
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Table"),
            ],
            psql_describe_index_rows(session),
        )
    } else {
        return None;
    };
    Some(write_single_row(stream, &columns, &rows))
}

#[cfg(test)]
pub(super) fn test_psql_describe_indexes_catalog_query() -> &'static str {
    psql_describe_indexes_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_indexes_catalog_query_schema_filter(
    canonical: &str,
) -> Option<String> {
    psql_describe_indexes_catalog_query_schema_filter(canonical)
}

#[cfg(test)]
pub(super) fn test_psql_describe_index_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    psql_describe_index_rows(session)
}

#[cfg(test)]
pub(super) fn test_psql_describe_index_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    psql_describe_index_verbose_rows(session)
}

pub(super) fn rename_index_in_session(
    session: &mut Session,
    old_name: &str,
    new_name: &str,
) -> Result<(), ErrorField> {
    if session.indexes.iter().any(|index| index.name == new_name)
        || session.tables.contains_key(new_name)
        || session.views.contains_key(new_name)
        || session.materialized_views.contains_key(new_name)
        || session.sequences.contains_key(new_name)
    {
        return Err(ErrorField {
            code: "42P07",
            message: "relation already exists",
            position: None,
        });
    }
    let Some(index) = session
        .indexes
        .iter_mut()
        .find(|index| index.name == old_name)
    else {
        return Err(ErrorField {
            code: "42704",
            message: "index does not exist",
            position: None,
        });
    };
    if index.primary_key || index.unique_constraint {
        return Err(ErrorField {
            code: "0A000",
            message: "cannot rename constraint-backed index with ALTER INDEX",
            position: None,
        });
    }
    let table_name = index.table.clone();
    index.name = new_name.to_string();
    session.dirty_indexes = true;
    session.mark_table_dirty(table_name);

    let old_target = CatalogCommentTarget::Index {
        index: old_name.to_string(),
    };
    if let Some(comment) = session.comments.remove(&old_target) {
        let new_target = CatalogCommentTarget::Index {
            index: new_name.to_string(),
        };
        session.comments.insert(new_target.clone(), comment);
        session.mark_comment_dirty(old_target);
        session.mark_comment_dirty(new_target);
    }

    session.persist_catalog_snapshot();
    Ok(())
}

pub(super) fn execute_index_ddl(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CreateIndex(create) => {
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if session
                .indexes
                .iter()
                .any(|index| index.name == create.name)
                || session.tables.contains_key(&create.name)
                || session.views.contains_key(&create.name)
                || session.materialized_views.contains_key(&create.name)
                || session.sequences.contains_key(&create.name)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P07",
                        message: "relation already exists",
                        position: None,
                    },
                );
            }
            let Some(table) = session.tables.get_mut(&create.table) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "relation does not exist",
                        position: None,
                    },
                );
            };
            if !table
                .columns
                .iter()
                .any(|column| column.def.name == create.column)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42703",
                        message: "column does not exist",
                        position: None,
                    },
                );
            }
            if create.unique {
                let mut candidate_indexes = session.indexes.clone();
                candidate_indexes.push(CatalogIndex {
                    name: create.name.clone(),
                    table: create.table.clone(),
                    column: create.column.clone(),
                    unique: true,
                    primary_key: false,
                    unique_constraint: false,
                });
                if let Err(error) = validate_unique_indexes(table, &candidate_indexes) {
                    return write_error(stream, &error);
                }
            }
            session.indexes.push(CatalogIndex {
                name: create.name,
                table: create.table.clone(),
                column: create.column,
                unique: create.unique,
                primary_key: false,
                unique_constraint: false,
            });
            session.dirty_indexes = true;
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE INDEX")
        }
        Command::RenameIndex(rename) => {
            if let Err(error) = rename_index_in_session(session, &rename.old_name, &rename.new_name)
            {
                return write_error(stream, &error);
            }
            write_command_complete(stream, "ALTER INDEX")
        }
        Command::DropIndex(drop) => {
            if !drop.if_exists {
                for name in &drop.names {
                    if !session.indexes.iter().any(|index| index.name == *name) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "index does not exist",
                                position: None,
                            },
                        );
                    }
                }
            }
            let old_index_count = session.indexes.len();
            let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
            session
                .indexes
                .retain(|index| !drop_names.contains(&index.name));
            session.dirty_indexes |= session.indexes.len() != old_index_count;
            if session.indexes.len() != old_index_count {
                for name in &drop.names {
                    let target = CatalogCommentTarget::Index {
                        index: name.clone(),
                    };
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                }
                let dropped_constraint_targets = session
                    .comments
                    .keys()
                    .filter(|target| match target {
                        CatalogCommentTarget::Constraint { constraint, .. } => {
                            drop_names.contains(constraint)
                        }
                        CatalogCommentTarget::Database { .. }
                        | CatalogCommentTarget::Role { .. }
                        | CatalogCommentTarget::Schema { .. }
                        | CatalogCommentTarget::Tablespace { .. }
                        | CatalogCommentTarget::Table { .. }
                        | CatalogCommentTarget::Column { .. }
                        | CatalogCommentTarget::Index { .. }
                        | CatalogCommentTarget::View { .. }
                        | CatalogCommentTarget::MaterializedView { .. }
                        | CatalogCommentTarget::Extension { .. }
                        | CatalogCommentTarget::Function { .. }
                        | CatalogCommentTarget::Sequence { .. }
                        | CatalogCommentTarget::Domain { .. }
                        | CatalogCommentTarget::Publication { .. }
                        | CatalogCommentTarget::Subscription { .. } => false,
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                for target in dropped_constraint_targets {
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                }
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP INDEX")
        }
        _ => unreachable!("index DDL executor called with an unrelated command"),
    }
}
