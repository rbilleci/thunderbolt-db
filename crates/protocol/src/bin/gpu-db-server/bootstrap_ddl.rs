// Legacy bootstrap catalog DDL ownership. This is not a product execution path.

use super::{
    write_command_complete, write_error, CatalogCommentTarget, Command, ErrorField, ReadWrite,
    Session,
};
use std::io;

pub(super) fn try_execute_bootstrap_ddl(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: &Command,
) -> Option<io::Result<()>> {
    match command {
        Command::CreateExtension(create) => {
            if create.name != "plpgsql" {
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "only the bootstrap plpgsql extension is supported",
                        position: None,
                    },
                ));
            }
            if create
                .schema
                .as_deref()
                .is_some_and(|schema| schema != "pg_catalog")
            {
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "plpgsql extension creation is only supported in pg_catalog",
                        position: None,
                    },
                ));
            }
            if !create.if_not_exists {
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "42710",
                        message: "extension \"plpgsql\" already exists",
                        position: None,
                    },
                ));
            }
            Some(write_command_complete(stream, "CREATE EXTENSION"))
        }
        Command::DropExtension(drop) => {
            if drop.name != "plpgsql" {
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "42704",
                        message: "extension does not exist",
                        position: None,
                    },
                ));
            }
            if !drop.if_exists {
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "cannot drop bootstrap extension \"plpgsql\"",
                        position: None,
                    },
                ));
            }
            Some(write_command_complete(stream, "DROP EXTENSION"))
        }
        Command::CreateSchema(create) => {
            if create.name != "public" {
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "only the public schema is supported",
                        position: None,
                    },
                ));
            }
            if session.public_schema_exists
                && !create.if_not_exists
                && !session.public_schema_implicit
            {
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "42P06",
                        message: "schema \"public\" already exists",
                        position: None,
                    },
                ));
            }
            session.public_schema_exists = true;
            session.public_schema_implicit = false;
            session.mark_schema_dirty();
            session.persist_catalog_snapshot();
            Some(write_command_complete(stream, "CREATE SCHEMA"))
        }
        Command::DropSchema(drop) => {
            if drop.name != "public" {
                if drop.if_exists {
                    return Some(write_command_complete(stream, "DROP SCHEMA"));
                }
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "3F000",
                        message: "schema does not exist",
                        position: None,
                    },
                ));
            }
            if !session.public_schema_exists {
                if drop.if_exists {
                    return Some(write_command_complete(stream, "DROP SCHEMA"));
                }
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "3F000",
                        message: "schema does not exist",
                        position: None,
                    },
                ));
            }
            if !session.tables.is_empty()
                || !session.views.is_empty()
                || !session.materialized_views.is_empty()
                || !session.functions.is_empty()
                || !session.sequences.is_empty()
                || !session.domains.is_empty()
                || !session.publications.is_empty()
                || !session.subscriptions.is_empty()
            {
                return Some(write_error(
                    stream,
                    &ErrorField {
                        code: "2BP01",
                        message: "cannot drop non-empty schema \"public\"",
                        position: None,
                    },
                ));
            }
            session.public_schema_exists = false;
            session.public_schema_implicit = false;
            session.schema_acl.clear();
            let target = CatalogCommentTarget::Schema {
                schema: "public".to_string(),
            };
            session.comments.remove(&target);
            session.mark_schema_dirty();
            session.mark_schema_acl_dirty();
            session.mark_comment_dirty(target);
            session.persist_catalog_snapshot();
            Some(write_command_complete(stream, "DROP SCHEMA"))
        }
        _ => None,
    }
}
