// Legacy domain DDL ownership. This is not a product execution path.

use super::{
    int4_column, schema_permission_error, sql_type_display_name, text_column,
    write_command_complete, write_error, write_single_row, CatalogCommentTarget, Command, Domain,
    ErrorField, ReadWrite, SchemaPrivilege, Session,
};
use std::collections::BTreeSet;
use std::io;

fn psql_list_domains_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", t.typname as \"name\", pg_catalog.format_type(t.typbasetype, t.typtypmod) as \"type\", (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type bt where c.oid = t.typcollation and bt.oid = t.typbasetype and t.typcollation <> bt.typcollation) as \"collation\", case when t.typnotnull then 'not null' end as \"nullable\", t.typdefault as \"default\", pg_catalog.array_to_string(array( select pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where t.oid = r.contypid ), ' ') as \"check\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where t.typtype = 'd' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_type_is_visible(t.oid) order by 1, 2"
}

fn psql_list_domains_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", t.typname as \"name\", pg_catalog.format_type(t.typbasetype, t.typtypmod) as \"type\", (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type bt where c.oid = t.typcollation and bt.oid = t.typbasetype and t.typcollation <> bt.typcollation) as \"collation\", case when t.typnotnull then 'not null' end as \"nullable\", t.typdefault as \"default\", pg_catalog.array_to_string(array( select pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where t.oid = r.contypid ), ' ') as \"check\", pg_catalog.array_to_string(t.typacl, e'\\n') as \"access privileges\", d.description as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace left join pg_catalog.pg_description d on d.classoid = t.tableoid and d.objoid = t.oid and d.objsubid = 0 where t.typtype = 'd' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_type_is_visible(t.oid) order by 1, 2"
}

fn catalog_domain_rows(session: &Session, verbose: bool) -> Vec<Vec<Option<String>>> {
    session
        .domains
        .values()
        .map(|domain| {
            let mut row = vec![
                Some("public".to_string()),
                Some(domain.name.clone()),
                Some(sql_type_display_name(domain.base_type).to_string()),
                None,
                None,
                None,
                None,
            ];
            if verbose {
                row.push(None);
                row.push(
                    session
                        .comments
                        .get(&CatalogCommentTarget::Domain {
                            domain: domain.name.clone(),
                        })
                        .cloned(),
                );
            }
            row
        })
        .collect()
}

fn catalog_domain_type_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    session
        .domains
        .values()
        .map(|domain| {
            vec![
                Some(domain.oid.to_string()),
                Some(domain.name.clone()),
                Some(domain.base_type.postgres_oid().to_string()),
                Some("d".to_string()),
            ]
        })
        .collect()
}

#[cfg(test)]
pub(super) fn test_psql_list_domains_catalog_query() -> &'static str {
    psql_list_domains_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_list_domains_verbose_catalog_query() -> &'static str {
    psql_list_domains_verbose_catalog_query()
}

pub(super) fn try_execute_domain_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == psql_list_domains_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Collation"),
                text_column("Nullable"),
                text_column("Default"),
                text_column("Check"),
            ],
            &catalog_domain_rows(session, false),
        ));
    }
    if canonical == psql_list_domains_verbose_catalog_query() {
        return Some(write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Collation"),
                text_column("Nullable"),
                text_column("Default"),
                text_column("Check"),
                text_column("Access privileges"),
                text_column("Description"),
            ],
            &catalog_domain_rows(session, true),
        ));
    }
    if canonical
        == "select oid, typname, typbasetype, typtype from pg_catalog.pg_type where typtype = 'd' order by typname"
    {
        return Some(write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("typname"),
                int4_column("typbasetype"),
                text_column("typtype"),
            ],
            &catalog_domain_type_rows(session),
        ));
    }
    None
}

pub(super) fn execute_domain_ddl(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CreateDomain(create) => {
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
                || session.domains.contains_key(&create.name)
            {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42710",
                        message: "type already exists",
                        position: None,
                    },
                );
            }
            let oid = session.next_relation_oid;
            session.next_relation_oid = match session.next_relation_oid.checked_add(1) {
                Some(next) => next,
                None => {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "54000",
                            message: "domain OID allocation exhausted",
                            position: None,
                        },
                    );
                }
            };
            let name = create.name;
            session.domains.insert(
                name.clone(),
                Domain {
                    oid,
                    name: name.clone(),
                    base_type: create.base_type,
                },
            );
            session.mark_domain_dirty(name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE DOMAIN")
        }
        Command::DropDomain(drop) => {
            let mut seen = BTreeSet::new();
            for name in &drop.domains {
                if !seen.insert(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42710",
                            message: "domain specified more than once",
                            position: None,
                        },
                    );
                }
                if !drop.if_exists && !session.domains.contains_key(name) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42704",
                            message: "domain does not exist",
                            position: None,
                        },
                    );
                }
                if session.tables.values().any(|table| {
                    table
                        .columns
                        .iter()
                        .any(|column| column.def.domain.as_deref() == Some(name.as_str()))
                }) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "2BP01",
                            message: "cannot drop domain because other objects depend on it",
                            position: None,
                        },
                    );
                }
            }
            for name in &drop.domains {
                if session.domains.remove(name).is_some() {
                    let target = CatalogCommentTarget::Domain {
                        domain: name.clone(),
                    };
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                }
                session.mark_domain_dirty(name.clone());
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP DOMAIN")
        }
        _ => unreachable!("domain DDL executor called with an unrelated command"),
    }
}
