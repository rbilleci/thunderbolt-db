// Legacy bootstrap catalog DDL ownership. This is not a product execution path.

use super::{
    bool_column, int4_column, schema_acl_array_display, schema_acl_display, text_column,
    write_command_complete, write_error, write_single_row, CatalogCommentTarget, Command,
    ErrorField, ReadWrite, Session, PG_EXTENSION_CLASS_OID, PLPGSQL_DESCRIPTION,
    PLPGSQL_EXTENSION_OID, PUBLIC_NAMESPACE_OID,
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

pub(super) fn try_execute_psql_schema_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == psql_describe_schemas_catalog_query() {
        return Some(write_single_row(
            stream,
            &[text_column("Name"), text_column("Owner")],
            &catalog_psql_describe_schema_rows(session),
        ));
    }
    if psql_describe_schemas_verbose_catalog_query_public_filter(canonical) {
        return Some(write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Access privileges"),
                text_column("Description"),
            ],
            &catalog_psql_describe_schema_verbose_rows(session),
        ));
    }
    None
}

pub(super) fn try_execute_namespace_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == pg_catalog_namespace_query() {
        return Some(write_single_row(
            stream,
            &[int4_column("oid"), text_column("nspname")],
            &pg_catalog_namespace_rows(session),
        ));
    }
    if canonical == pg_catalog_namespace_acl_query() {
        return Some(write_single_row(
            stream,
            &[text_column("nspname"), text_column("nspacl")],
            &pg_catalog_namespace_acl_rows(session),
        ));
    }
    None
}

pub(super) fn try_execute_schema_pg_dump_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical
        == "select n.tableoid, n.oid, n.nspname, n.nspowner, n.nspacl, acldefault('n', n.nspowner) as acldefault from pg_namespace n"
    {
        return Some(write_single_row(
            stream,
            &[
                int4_column("tableoid"),
                int4_column("oid"),
                text_column("nspname"),
                int4_column("nspowner"),
                text_column("nspacl"),
                text_column("acldefault"),
            ],
            &[
                vec![
                    Some("2615".to_string()),
                    Some("11".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("10".to_string()),
                    None,
                    None,
                ],
                vec![
                    Some("2615".to_string()),
                    Some(PUBLIC_NAMESPACE_OID.to_string()),
                    Some("public".to_string()),
                    Some("10".to_string()),
                    schema_acl_array_display(session),
                    Some("{postgres=UC/postgres,=U/postgres}".to_string()),
                ],
            ],
        ));
    }
    if is_pg_dump_public_namespace_oid_lookup_query(canonical) {
        return Some(write_single_row(
            stream,
            &[int4_column("oid")],
            &[vec![Some(PUBLIC_NAMESPACE_OID.to_string())]],
        ));
    }
    None
}

pub(super) fn try_execute_information_schema_schemata_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical != information_schema_schemata_query() {
        return None;
    }
    Some(write_single_row(
        stream,
        &[text_column("schema_name"), text_column("schema_owner")],
        &information_schema_schemata_rows(session),
    ))
}

fn psql_describe_schemas_catalog_query() -> &'static str {
    "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\" from pg_catalog.pg_namespace n where n.nspname !~ '^pg_' and n.nspname <> 'information_schema' order by 1"
}

fn psql_describe_schemas_verbose_catalog_query_public_filter(canonical: &str) -> bool {
    canonical
        == "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\", pg_catalog.array_to_string(n.nspacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(n.oid, 'pg_namespace') as \"description\" from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1"
}

fn catalog_psql_describe_schema_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
    ]]
}

fn catalog_psql_describe_schema_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
        schema_acl_display(session),
        session
            .comments
            .get(&CatalogCommentTarget::Schema {
                schema: "public".to_string(),
            })
            .cloned(),
    ]]
}

fn pg_catalog_namespace_query() -> &'static str {
    "select oid, nspname from pg_catalog.pg_namespace where nspname = 'public' order by oid"
}

fn pg_catalog_namespace_acl_query() -> &'static str {
    "select n.nspname, n.nspacl from pg_catalog.pg_namespace n where n.nspname = 'public' order by n.nspname"
}

fn pg_catalog_namespace_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some(PUBLIC_NAMESPACE_OID.to_string()),
        Some("public".to_string()),
    ]]
}

fn pg_catalog_namespace_acl_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        schema_acl_display(session),
    ]]
}

fn is_pg_dump_public_namespace_oid_lookup_query(canonical: &str) -> bool {
    canonical
        == "select oid from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default"
}

fn information_schema_schemata_query() -> &'static str {
    "select schema_name, schema_owner from information_schema.schemata where schema_name = 'public' order by schema_name"
}

fn information_schema_schemata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
    ]]
}

#[cfg(test)]
pub(super) fn test_psql_describe_schemas_catalog_query() -> &'static str {
    psql_describe_schemas_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_schemas_verbose_catalog_query_public_filter(
    canonical: &str,
) -> bool {
    psql_describe_schemas_verbose_catalog_query_public_filter(canonical)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_schema_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_schema_rows(session)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_schema_verbose_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_schema_verbose_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_catalog_namespace_query() -> &'static str {
    pg_catalog_namespace_query()
}

#[cfg(test)]
pub(super) fn test_pg_catalog_namespace_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_catalog_namespace_rows(session)
}

#[cfg(test)]
pub(super) fn test_is_pg_dump_public_namespace_oid_lookup_query(canonical: &str) -> bool {
    is_pg_dump_public_namespace_oid_lookup_query(canonical)
}

#[cfg(test)]
pub(super) fn test_information_schema_schemata_query() -> &'static str {
    information_schema_schemata_query()
}

#[cfg(test)]
pub(super) fn test_information_schema_schemata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    information_schema_schemata_rows(session)
}
