// Legacy publication/subscription catalog ownership. This is not a product execution path.

use super::{
    schema_permission_error, write_command_complete, write_error, CatalogCommentTarget, Command,
    ErrorField, Publication, PublicationTarget, ReadWrite, SchemaPrivilege, Session, Subscription,
};
use std::collections::BTreeSet;
use std::io;

fn publication_table_target_error(session: &Session, table: &str) -> Option<ErrorField> {
    if session.views.contains_key(table)
        || session.materialized_views.contains_key(table)
        || session.sequences.contains_key(table)
    {
        return Some(ErrorField {
            code: "42809",
            message: "relation is not a table",
            position: None,
        });
    }
    if !session.tables.contains_key(table) {
        return Some(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    }
    None
}

fn create_publication(
    session: &mut Session,
    name: String,
    target: PublicationTarget,
) -> Result<(), ErrorField> {
    if session.publications.contains_key(&name) {
        return Err(ErrorField {
            code: "42710",
            message: "publication already exists",
            position: None,
        });
    }
    let (all_tables, tables) = match target {
        PublicationTarget::AllTables => (true, Vec::new()),
        PublicationTarget::Tables(tables) => {
            let mut seen = BTreeSet::new();
            for table in &tables {
                if !seen.insert(table.clone()) {
                    return Err(ErrorField {
                        code: "42710",
                        message: "publication table specified more than once",
                        position: None,
                    });
                }
                if let Some(error) = publication_table_target_error(session, table) {
                    return Err(error);
                }
            }
            (false, tables)
        }
    };
    let oid = session.next_relation_oid;
    session.next_relation_oid = session.next_relation_oid.checked_add(1).ok_or(ErrorField {
        code: "54000",
        message: "publication OID allocation exhausted",
        position: None,
    })?;
    session.publications.insert(
        name.clone(),
        Publication {
            oid,
            name: name.clone(),
            all_tables,
            tables,
        },
    );
    session.mark_publication_dirty(name);
    Ok(())
}

fn drop_publication(
    session: &mut Session,
    names: &[String],
    if_exists: bool,
) -> Result<(), ErrorField> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(ErrorField {
                code: "42710",
                message: "publication specified more than once",
                position: None,
            });
        }
        if !if_exists && !session.publications.contains_key(name) {
            return Err(ErrorField {
                code: "42704",
                message: "publication does not exist",
                position: None,
            });
        }
    }
    for name in names {
        session.publications.remove(name);
        let target = CatalogCommentTarget::Publication {
            publication: name.clone(),
        };
        session.comments.remove(&target);
        session.mark_comment_dirty(target);
        session.mark_publication_dirty(name.clone());
    }
    Ok(())
}

fn create_subscription(
    session: &mut Session,
    name: String,
    connection: String,
    publications: Vec<String>,
) -> Result<(), ErrorField> {
    if session.subscriptions.contains_key(&name) {
        return Err(ErrorField {
            code: "42710",
            message: "subscription already exists",
            position: None,
        });
    }
    let mut seen = BTreeSet::new();
    for publication in &publications {
        if !seen.insert(publication.clone()) {
            return Err(ErrorField {
                code: "42710",
                message: "subscription publication specified more than once",
                position: None,
            });
        }
        if !session.publications.contains_key(publication) {
            return Err(ErrorField {
                code: "42704",
                message: "publication does not exist",
                position: None,
            });
        }
    }
    let oid = session.next_relation_oid;
    session.next_relation_oid = session.next_relation_oid.checked_add(1).ok_or(ErrorField {
        code: "54000",
        message: "subscription OID allocation exhausted",
        position: None,
    })?;
    session.subscriptions.insert(
        name.clone(),
        Subscription {
            oid,
            name: name.clone(),
            connection,
            publications,
            enabled: false,
        },
    );
    session.mark_subscription_dirty(name);
    Ok(())
}

fn drop_subscription(
    session: &mut Session,
    names: &[String],
    if_exists: bool,
) -> Result<(), ErrorField> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(ErrorField {
                code: "42710",
                message: "subscription specified more than once",
                position: None,
            });
        }
        if !if_exists && !session.subscriptions.contains_key(name) {
            return Err(ErrorField {
                code: "42704",
                message: "subscription does not exist",
                position: None,
            });
        }
    }
    for name in names {
        session.subscriptions.remove(name);
        let target = CatalogCommentTarget::Subscription {
            subscription: name.clone(),
        };
        session.comments.remove(&target);
        session.mark_comment_dirty(target);
        session.mark_subscription_dirty(name.clone());
    }
    Ok(())
}

pub(super) fn execute_replication_catalog_command(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CreatePublication(create) => {
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if let Err(error) = create_publication(session, create.name, create.target) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE PUBLICATION")
        }
        Command::DropPublication(drop) => {
            if let Err(error) = drop_publication(session, &drop.names, drop.if_exists) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP PUBLICATION")
        }
        Command::CreateSubscription(create) => {
            if let Some(error) = schema_permission_error(session, "public", SchemaPrivilege::Create)
            {
                return write_error(stream, &error);
            }
            if let Err(error) =
                create_subscription(session, create.name, create.connection, create.publications)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE SUBSCRIPTION")
        }
        Command::DropSubscription(drop) => {
            if let Err(error) = drop_subscription(session, &drop.names, drop.if_exists) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP SUBSCRIPTION")
        }
        _ => unreachable!("replication catalog executor called with an unrelated command"),
    }
}
