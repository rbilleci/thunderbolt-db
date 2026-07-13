// Legacy bootstrap catalog DDL ownership. This is not a product execution path.

use super::{
    bool_column, int4_column, text_column, write_command_complete, write_error, write_single_row,
    CatalogCommentTarget, Command, ErrorField, ReadWrite, Session, PG_EXTENSION_CLASS_OID,
    PLPGSQL_DESCRIPTION, PLPGSQL_EXTENSION_OID,
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

pub(super) fn try_execute_extension_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical != psql_list_extensions_catalog_query() {
        return None;
    }
    Some(write_single_row(
        stream,
        &[
            text_column("Name"),
            text_column("Version"),
            text_column("Schema"),
            text_column("Description"),
        ],
        &catalog_psql_extension_rows(session),
    ))
}

pub(super) fn try_execute_extension_pg_dump_catalog_query(
    stream: &mut dyn ReadWrite,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical
        != "select x.tableoid, x.oid, x.extname, n.nspname, x.extrelocatable, x.extversion, x.extconfig, x.extcondition from pg_extension x join pg_namespace n on n.oid = x.extnamespace"
    {
        return None;
    }
    Some(write_single_row(
        stream,
        &[
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("extname"),
            text_column("nspname"),
            bool_column("extrelocatable"),
            text_column("extversion"),
            text_column("extconfig"),
            text_column("extcondition"),
        ],
        &catalog_extension_discovery_rows(),
    ))
}

fn psql_list_extensions_catalog_query() -> &'static str {
    "select e.extname as \"name\", e.extversion as \"version\", n.nspname as \"schema\", c.description as \"description\" from pg_catalog.pg_extension e left join pg_catalog.pg_namespace n on n.oid = e.extnamespace left join pg_catalog.pg_description c on c.objoid = e.oid and c.classoid = 'pg_catalog.pg_extension'::pg_catalog.regclass order by 1"
}

fn bootstrap_extension_description(session: &Session) -> String {
    session
        .comments
        .get(&CatalogCommentTarget::Extension {
            extension: "plpgsql".to_string(),
        })
        .cloned()
        .unwrap_or_else(|| PLPGSQL_DESCRIPTION.to_string())
}

fn catalog_psql_extension_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("plpgsql".to_string()),
        Some("1.0".to_string()),
        Some("pg_catalog".to_string()),
        Some(bootstrap_extension_description(session)),
    ]]
}

fn catalog_extension_discovery_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some(PG_EXTENSION_CLASS_OID.to_string()),
        Some(PLPGSQL_EXTENSION_OID.to_string()),
        Some("plpgsql".to_string()),
        Some("pg_catalog".to_string()),
        Some("f".to_string()),
        Some("1.0".to_string()),
        None,
        None,
    ]]
}

#[cfg(test)]
pub(super) fn test_catalog_psql_extension_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_extension_rows(session)
}

#[cfg(test)]
pub(super) fn test_psql_list_extensions_catalog_query() -> &'static str {
    psql_list_extensions_catalog_query()
}
