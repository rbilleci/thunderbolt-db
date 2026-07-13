// Legacy catalog-comment ownership. This is not a product execution path.

use super::{
    database_exists, role_exists, shared_catalog_contains_live_index,
    shared_catalog_contains_sequence, shared_catalog_contains_table,
    shared_catalog_contains_table_constraint, shared_catalog_contains_view, tablespace_exists,
    write_command_complete, write_error, CatalogCommentTarget, ErrorField, ReadWrite, Session,
};
use gpu_db_protocol::{Command, CommentTarget};
use std::io;

pub(super) fn execute_catalog_comment(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CommentOn(comment) => {
            let target = match comment.target {
                CommentTarget::Database { database } => {
                    if !database_exists(session, &database) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "3D000",
                                message: "database does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Database { database }
                }
                CommentTarget::Role { role } => {
                    if !role_exists(session, &role) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "role does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Role { role }
                }
                CommentTarget::Schema { schema } => {
                    if schema != "public" {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "3F000",
                                message: "schema does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Schema { schema }
                }
                CommentTarget::Tablespace { tablespace } => {
                    if !tablespace_exists(session, &tablespace) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "tablespace does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Tablespace { tablespace }
                }
                CommentTarget::Table { table } => {
                    if !session.tables.contains_key(&table) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "relation does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Table { table }
                }
                CommentTarget::Column { table, column } => {
                    let Some(table_ref) = session.tables.get(&table) else {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "relation does not exist",
                                position: None,
                            },
                        );
                    };
                    let Some(column_ref) = table_ref
                        .columns
                        .iter()
                        .find(|candidate| candidate.def.name == column)
                    else {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42703",
                                message: "column does not exist",
                                position: None,
                            },
                        );
                    };
                    CatalogCommentTarget::Column {
                        table,
                        attnum: column_ref.attnum,
                    }
                }
                CommentTarget::Index { index } => {
                    let exists = session.indexes.iter().any(|candidate| {
                        candidate.name == index && session.tables.contains_key(&candidate.table)
                    }) || (session.shared_catalog
                        && shared_catalog_contains_live_index(&index));
                    if !exists {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "index does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Index { index }
                }
                CommentTarget::View { view } => {
                    let exists = session.views.contains_key(&view)
                        || (session.shared_catalog && shared_catalog_contains_view(&view));
                    if !exists {
                        if session.tables.contains_key(&view)
                            || (session.shared_catalog && shared_catalog_contains_table(&view))
                            || session.sequences.contains_key(&view)
                            || (session.shared_catalog && shared_catalog_contains_sequence(&view))
                        {
                            return write_error(
                                stream,
                                &ErrorField {
                                    code: "42809",
                                    message: "relation is not a view",
                                    position: None,
                                },
                            );
                        }
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "view does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::View { view }
                }
                CommentTarget::MaterializedView { materialized_view } => {
                    let exists = session.materialized_views.contains_key(&materialized_view);
                    if !exists {
                        if session.tables.contains_key(&materialized_view)
                            || session.views.contains_key(&materialized_view)
                            || session.sequences.contains_key(&materialized_view)
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
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "materialized view does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::MaterializedView { materialized_view }
                }
                CommentTarget::Function { function } => {
                    if !session.functions.contains_key(&function) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42883",
                                message: "function does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Function { function }
                }
                CommentTarget::Extension { extension } => {
                    if extension != "plpgsql" {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "extension does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Extension { extension }
                }
                CommentTarget::Sequence { sequence } => {
                    let exists = session.sequences.contains_key(&sequence)
                        || (session.shared_catalog && shared_catalog_contains_sequence(&sequence));
                    if !exists {
                        if session.tables.contains_key(&sequence)
                            || session.views.contains_key(&sequence)
                            || (session.shared_catalog
                                && (shared_catalog_contains_table(&sequence)
                                    || shared_catalog_contains_view(&sequence)))
                        {
                            return write_error(
                                stream,
                                &ErrorField {
                                    code: "42809",
                                    message: "relation is not a sequence",
                                    position: None,
                                },
                            );
                        }
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "sequence does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Sequence { sequence }
                }
                CommentTarget::Domain { domain } => {
                    let exists = session.domains.contains_key(&domain);
                    if !exists {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "domain does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Domain { domain }
                }
                CommentTarget::Publication { publication } => {
                    if !session.publications.contains_key(&publication) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "publication does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Publication { publication }
                }
                CommentTarget::Subscription { subscription } => {
                    if !session.subscriptions.contains_key(&subscription) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "subscription does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Subscription { subscription }
                }
                CommentTarget::Constraint { table, constraint } => {
                    let table_exists = session.tables.contains_key(&table)
                        || (session.shared_catalog && shared_catalog_contains_table(&table));
                    if !table_exists {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42P01",
                                message: "relation does not exist",
                                position: None,
                            },
                        );
                    }
                    let exists = session.indexes.iter().any(|candidate| {
                        candidate.table == table
                            && candidate.name == constraint
                            && (candidate.primary_key || candidate.unique_constraint)
                    }) || session.tables.get(&table).is_some_and(|table| {
                        table
                            .check_constraints
                            .iter()
                            .any(|candidate| candidate.name == constraint)
                            || table
                                .foreign_keys
                                .iter()
                                .any(|candidate| candidate.name == constraint)
                    }) || (session.shared_catalog
                        && shared_catalog_contains_table_constraint(&table, &constraint));
                    if !exists {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42704",
                                message: "constraint does not exist",
                                position: None,
                            },
                        );
                    }
                    CatalogCommentTarget::Constraint { table, constraint }
                }
            };
            if let Some(value) = comment.comment {
                session.comments.insert(target.clone(), value);
            } else {
                session.comments.remove(&target);
            }
            session.mark_comment_dirty(target);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "COMMENT")
        }
        _ => unreachable!("catalog-comment executor received an unrelated command"),
    }
}
