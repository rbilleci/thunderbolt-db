// Legacy domain DDL ownership. This is not a product execution path.

use super::{
    schema_permission_error, write_command_complete, write_error, CatalogCommentTarget, Command,
    Domain, ErrorField, ReadWrite, SchemaPrivilege, Session,
};
use std::collections::BTreeSet;
use std::io;

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
