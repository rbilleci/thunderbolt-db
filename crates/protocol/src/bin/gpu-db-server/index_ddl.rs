// Legacy index DDL ownership. This is not a product execution path.

use super::{
    rename_index_in_session, schema_permission_error, validate_unique_indexes,
    write_command_complete, write_error, CatalogCommentTarget, CatalogIndex, Command, ErrorField,
    ReadWrite, SchemaPrivilege, Session,
};
use std::collections::BTreeSet;
use std::io;

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
