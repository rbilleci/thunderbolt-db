// Legacy publication/subscription catalog ownership. This is not a product execution path.

use super::{
    create_publication, create_subscription, drop_publication, drop_subscription,
    schema_permission_error, write_command_complete, write_error, Command, ReadWrite,
    SchemaPrivilege, Session,
};
use std::io;

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
