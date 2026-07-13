// Legacy view DDL ownership. This is not a product execution path.

use super::{
    execute_select_result, materialize_select_rows, schema_permission_error, text_column,
    write_command_complete, write_error, write_single_row, CatalogCommentTarget, Command,
    ErrorField, MaterializedView, ReadWrite, SchemaPrivilege, Session, View,
};
use std::collections::BTreeSet;
use std::io;

fn psql_describe_views_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('v','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_views_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('v','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_materialized_views_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('m','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_materialized_views_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('m','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_view_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .views
        .values()
        .map(|view| {
            vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("view".to_string()),
                Some("postgres".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn psql_describe_view_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .views
        .values()
        .map(|view| {
            vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("view".to_string()),
                Some("postgres".to_string()),
                Some("permanent".to_string()),
                None,
                session
                    .comments
                    .get(&CatalogCommentTarget::View {
                        view: view.name.clone(),
                    })
                    .cloned(),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn psql_describe_materialized_view_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .materialized_views
        .values()
        .map(|view| {
            vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("materialized view".to_string()),
                Some("postgres".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn psql_describe_materialized_view_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .materialized_views
        .values()
        .map(|view| {
            vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("materialized view".to_string()),
                Some("postgres".to_string()),
                Some("permanent".to_string()),
                Some("heap".to_string()),
                Some("0 bytes".to_string()),
                session
                    .comments
                    .get(&CatalogCommentTarget::MaterializedView {
                        materialized_view: view.name.clone(),
                    })
                    .cloned(),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

pub(super) fn try_execute_view_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    let (columns, rows) = if canonical == psql_describe_views_catalog_query() {
        (
            vec![
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            psql_describe_view_rows(session),
        )
    } else if canonical == psql_describe_views_verbose_catalog_query() {
        (
            vec![
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Size"),
                text_column("Description"),
            ],
            psql_describe_view_verbose_rows(session),
        )
    } else if canonical == psql_describe_materialized_views_catalog_query() {
        (
            vec![
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            psql_describe_materialized_view_rows(session),
        )
    } else if canonical == psql_describe_materialized_views_verbose_catalog_query() {
        (
            vec![
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            psql_describe_materialized_view_verbose_rows(session),
        )
    } else {
        return None;
    };
    Some(write_single_row(stream, &columns, &rows))
}

#[cfg(test)]
pub(super) fn test_psql_describe_views_catalog_query() -> &'static str {
    psql_describe_views_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_views_verbose_catalog_query() -> &'static str {
    psql_describe_views_verbose_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_materialized_views_catalog_query() -> &'static str {
    psql_describe_materialized_views_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_materialized_views_verbose_catalog_query() -> &'static str {
    psql_describe_materialized_views_verbose_catalog_query()
}

fn session_view_depends_on(session: &Session, view: &str, target: &str) -> bool {
    let mut seen = BTreeSet::new();
    session_view_depends_on_inner(session, view, target, &mut seen)
}

fn session_view_depends_on_inner(
    session: &Session,
    view: &str,
    target: &str,
    seen: &mut BTreeSet<String>,
) -> bool {
    if view == target {
        return true;
    }
    if !seen.insert(view.to_string()) {
        return false;
    }
    let Some(view) = session.views.get(view) else {
        return false;
    };
    session_view_depends_on_inner(session, &view.query.table, target, seen)
}

fn session_view_has_dependents(session: &Session, view: &str) -> bool {
    session
        .views
        .keys()
        .any(|candidate| candidate != view && session_view_depends_on(session, candidate, view))
}

pub(super) fn execute_view_ddl(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CreateView(create) => {
            if !session.public_schema_exists {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "3F000",
                        message: "schema does not exist",
                        position: None,
                    },
                );
            }
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if session.tables.contains_key(&create.name)
                || session.materialized_views.contains_key(&create.name)
                || session.sequences.contains_key(&create.name)
                || (!create.or_replace && session.views.contains_key(&create.name))
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
            if session.materialized_views.contains_key(&create.query.table) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "views over materialized views are unsupported",
                        position: None,
                    },
                );
            }
            if create.or_replace && session_view_has_dependents(session, &create.name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "2BP01",
                        message: "cannot replace view because another view depends on it",
                        position: None,
                    },
                );
            }
            if session.views.contains_key(&create.query.table)
                && session_view_depends_on(session, &create.query.table, &create.name)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "view dependency cycle is unsupported",
                        position: None,
                    },
                );
            }
            if let Err(error) = execute_select_result(session, &create.query) {
                return write_error(stream, &error);
            }
            let oid = if let Some(existing) = session.views.get(&create.name) {
                existing.oid
            } else {
                let oid = session.next_relation_oid;
                session.next_relation_oid = match session.next_relation_oid.checked_add(1) {
                    Some(next) => next,
                    None => {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "54000",
                                message: "relation OID allocation exhausted",
                                position: None,
                            },
                        );
                    }
                };
                oid
            };
            let name = create.name;
            session.views.insert(
                name.clone(),
                View {
                    oid,
                    name: name.clone(),
                    query: create.query,
                    definition: create.definition,
                },
            );
            session.mark_view_dirty(name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE VIEW")
        }
        Command::CreateMaterializedView(create) => {
            if !session.public_schema_exists {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "3F000",
                        message: "schema does not exist",
                        position: None,
                    },
                );
            }
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if session.tables.contains_key(&create.name)
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
            if session.views.contains_key(&create.query.table)
                || session.materialized_views.contains_key(&create.query.table)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "materialized views over views are unsupported",
                        position: None,
                    },
                );
            }
            let result = match execute_select_result(session, &create.query) {
                Ok(result) => result,
                Err(error) => return write_error(stream, &error),
            };
            let oid = session.next_relation_oid;
            session.next_relation_oid = match session.next_relation_oid.checked_add(1) {
                Some(next) => next,
                None => {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "54000",
                            message: "relation OID allocation exhausted",
                            position: None,
                        },
                    );
                }
            };
            let (columns, mut rows) = match materialize_select_rows(result, None) {
                Ok(materialized) => materialized,
                Err(error) => return write_error(stream, &error),
            };
            if !create.with_data {
                rows.clear();
            }
            let name = create.name;
            session.materialized_views.insert(
                name.clone(),
                MaterializedView {
                    oid,
                    name: name.clone(),
                    query: create.query,
                    definition: create.definition,
                    columns,
                    rows,
                },
            );
            session.mark_materialized_view_dirty(name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "SELECT 0")
        }
        Command::RefreshMaterializedView(refresh) => {
            if session.tables.contains_key(&refresh.name)
                || session.views.contains_key(&refresh.name)
                || session.sequences.contains_key(&refresh.name)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42809",
                        message: "relation is not a materialized view",
                        position: None,
                    },
                );
            }
            let Some(existing) = session.materialized_views.get(&refresh.name) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "materialized view does not exist",
                        position: None,
                    },
                );
            };
            let query = existing.query.clone();
            let expected_columns = existing.columns.clone();
            let result = match execute_select_result(session, &query) {
                Ok(result) => result,
                Err(error) => return write_error(stream, &error),
            };
            let (_, rows) = match materialize_select_rows(result, Some(&expected_columns)) {
                Ok(materialized) => materialized,
                Err(error) => return write_error(stream, &error),
            };
            let view = session
                .materialized_views
                .get_mut(&refresh.name)
                .expect("materialized view existence validated");
            view.rows = rows;
            session.mark_materialized_view_dirty(refresh.name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "REFRESH MATERIALIZED VIEW")
        }
        Command::RenameView(rename) => {
            if session.tables.contains_key(&rename.old_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42809",
                        message: "relation is not a view",
                        position: None,
                    },
                );
            }
            if !session.views.contains_key(&rename.old_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "view does not exist",
                        position: None,
                    },
                );
            }
            if session.tables.contains_key(&rename.new_name)
                || session.views.contains_key(&rename.new_name)
                || session.materialized_views.contains_key(&rename.new_name)
                || session.sequences.contains_key(&rename.new_name)
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
            if session_view_has_dependents(session, &rename.old_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "2BP01",
                        message: "cannot rename view because another view depends on it",
                        position: None,
                    },
                );
            }
            let mut view = session
                .views
                .remove(&rename.old_name)
                .expect("view existence validated");
            view.name = rename.new_name.clone();
            session.views.insert(rename.new_name.clone(), view);
            session.mark_view_dirty(rename.old_name.clone());
            session.mark_view_dirty(rename.new_name.clone());
            if let Some(acl) = session.table_acls.remove(&rename.old_name) {
                session.table_acls.insert(rename.new_name.clone(), acl);
                session.mark_table_acl_dirty(rename.old_name.clone());
                session.mark_table_acl_dirty(rename.new_name.clone());
            }
            let old_target = CatalogCommentTarget::View {
                view: rename.old_name,
            };
            if let Some(comment) = session.comments.remove(&old_target) {
                session.mark_comment_dirty(old_target);
                let new_target = CatalogCommentTarget::View {
                    view: rename.new_name,
                };
                session.comments.insert(new_target.clone(), comment);
                session.mark_comment_dirty(new_target);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER VIEW")
        }
        Command::RenameMaterializedView(rename) => {
            if session.tables.contains_key(&rename.old_name)
                || session.views.contains_key(&rename.old_name)
                || session.sequences.contains_key(&rename.old_name)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42809",
                        message: "relation is not a materialized view",
                        position: None,
                    },
                );
            }
            if !session.materialized_views.contains_key(&rename.old_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P01",
                        message: "materialized view does not exist",
                        position: None,
                    },
                );
            }
            if session.tables.contains_key(&rename.new_name)
                || session.views.contains_key(&rename.new_name)
                || session.materialized_views.contains_key(&rename.new_name)
                || session.sequences.contains_key(&rename.new_name)
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
            let mut view = session
                .materialized_views
                .remove(&rename.old_name)
                .expect("materialized view existence validated");
            view.name = rename.new_name.clone();
            session
                .materialized_views
                .insert(rename.new_name.clone(), view);
            session.mark_materialized_view_dirty(rename.old_name.clone());
            session.mark_materialized_view_dirty(rename.new_name.clone());
            if let Some(acl) = session.table_acls.remove(&rename.old_name) {
                session.table_acls.insert(rename.new_name.clone(), acl);
                session.mark_table_acl_dirty(rename.old_name.clone());
                session.mark_table_acl_dirty(rename.new_name.clone());
            }
            let old_target = CatalogCommentTarget::MaterializedView {
                materialized_view: rename.old_name,
            };
            if let Some(comment) = session.comments.remove(&old_target) {
                session.mark_comment_dirty(old_target);
                let new_target = CatalogCommentTarget::MaterializedView {
                    materialized_view: rename.new_name,
                };
                session.comments.insert(new_target.clone(), comment);
                session.mark_comment_dirty(new_target);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER MATERIALIZED VIEW")
        }
        Command::DropView(drop) => {
            let mut seen = BTreeSet::new();
            let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
            for name in &drop.names {
                if !seen.insert(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42710",
                            message: "view specified more than once",
                            position: None,
                        },
                    );
                }
                if session.tables.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42809",
                            message: "relation is not a view",
                            position: None,
                        },
                    );
                }
                if session.sequences.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42809",
                            message: "relation is not a view",
                            position: None,
                        },
                    );
                }
                if session.materialized_views.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42809",
                            message: "relation is not a view",
                            position: None,
                        },
                    );
                }
                if !drop.if_exists && !session.views.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42P01",
                            message: "view does not exist",
                            position: None,
                        },
                    );
                }
                if session.views.keys().any(|candidate| {
                    !drop_names.contains(candidate)
                        && session_view_depends_on(session, candidate, name)
                }) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "2BP01",
                            message: "cannot drop view because another view depends on it",
                            position: None,
                        },
                    );
                }
            }
            for name in &drop.names {
                if session.views.remove(name).is_some() {
                    session.table_acls.remove(name);
                    session.mark_table_acl_dirty(name.clone());
                    let target = CatalogCommentTarget::View { view: name.clone() };
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                }
                session.mark_view_dirty(name.clone());
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP VIEW")
        }
        Command::DropMaterializedView(drop) => {
            let mut seen = BTreeSet::new();
            for name in &drop.names {
                if !seen.insert(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42710",
                            message: "materialized view specified more than once",
                            position: None,
                        },
                    );
                }
                if session.tables.contains_key(name)
                    || session.views.contains_key(name)
                    || session.sequences.contains_key(name)
                {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42809",
                            message: "relation is not a materialized view",
                            position: None,
                        },
                    );
                }
                if !drop.if_exists && !session.materialized_views.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42P01",
                            message: "materialized view does not exist",
                            position: None,
                        },
                    );
                }
            }
            for name in &drop.names {
                if session.materialized_views.remove(name).is_some() {
                    session.table_acls.remove(name);
                    session.mark_table_acl_dirty(name.clone());
                    let target = CatalogCommentTarget::MaterializedView {
                        materialized_view: name.clone(),
                    };
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                }
                session.mark_materialized_view_dirty(name.clone());
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP MATERIALIZED VIEW")
        }
        _ => unreachable!("view DDL executor called with an unrelated command"),
    }
}
