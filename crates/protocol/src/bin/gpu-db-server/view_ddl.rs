// Legacy view DDL ownership. This is not a product execution path.

use super::{
    execute_select_result, materialize_select_rows, schema_permission_error,
    session_view_depends_on, session_view_has_dependents, write_command_complete, write_error,
    CatalogCommentTarget, Command, ErrorField, MaterializedView, ReadWrite, SchemaPrivilege,
    Session, View,
};
use std::collections::BTreeSet;
use std::io;

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
