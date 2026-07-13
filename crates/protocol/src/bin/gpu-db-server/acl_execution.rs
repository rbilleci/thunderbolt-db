// Legacy ACL mutation ownership. This is not a product execution path.

use super::{
    grant_database_acl, grant_default_table_acl, grant_function_acl, grant_relation_acl,
    grant_schema_acl, grant_tablespace_acl, revoke_database_acl, revoke_default_table_acl,
    revoke_function_acl, revoke_relation_acl, revoke_schema_acl, revoke_tablespace_acl,
    write_command_complete, write_error, Command, ReadWrite, Session,
};
use std::io;

pub(super) fn execute_acl_command(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::GrantTable(grant) => {
            if let Err(error) = grant_relation_acl(
                session,
                &grant.relation,
                grant.kind,
                &grant.grantee,
                &grant.privileges,
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "GRANT")
        }
        Command::RevokeTable(revoke) => {
            if let Err(error) = revoke_relation_acl(
                session,
                &revoke.relation,
                revoke.kind,
                &revoke.grantee,
                &revoke.privileges,
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "REVOKE")
        }
        Command::GrantSchema(grant) => {
            if let Err(error) =
                grant_schema_acl(session, &grant.schema, &grant.grantee, &grant.privileges)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "GRANT")
        }
        Command::RevokeSchema(revoke) => {
            if let Err(error) =
                revoke_schema_acl(session, &revoke.schema, &revoke.grantee, &revoke.privileges)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "REVOKE")
        }
        Command::GrantDatabase(grant) => {
            if let Err(error) =
                grant_database_acl(session, &grant.database, &grant.grantee, &grant.privileges)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "GRANT")
        }
        Command::RevokeDatabase(revoke) => {
            if let Err(error) = revoke_database_acl(
                session,
                &revoke.database,
                &revoke.grantee,
                &revoke.privileges,
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "REVOKE")
        }
        Command::GrantTablespace(grant) => {
            if let Err(error) = grant_tablespace_acl(
                session,
                &grant.tablespace,
                &grant.grantee,
                &grant.privileges,
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "GRANT")
        }
        Command::RevokeTablespace(revoke) => {
            if let Err(error) = revoke_tablespace_acl(
                session,
                &revoke.tablespace,
                &revoke.grantee,
                &revoke.privileges,
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "REVOKE")
        }
        Command::GrantFunction(grant) => {
            if let Err(error) =
                grant_function_acl(session, &grant.function, &grant.grantee, &grant.privileges)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "GRANT")
        }
        Command::RevokeFunction(revoke) => {
            if let Err(error) = revoke_function_acl(
                session,
                &revoke.function,
                &revoke.grantee,
                &revoke.privileges,
            ) {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "REVOKE")
        }
        Command::GrantDefaultTablePrivileges(grant) => {
            if let Err(error) = grant_default_table_acl(session, &grant.grantee, &grant.privileges)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER DEFAULT PRIVILEGES")
        }
        Command::RevokeDefaultTablePrivileges(revoke) => {
            if let Err(error) =
                revoke_default_table_acl(session, &revoke.grantee, &revoke.privileges)
            {
                return write_error(stream, &error);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER DEFAULT PRIVILEGES")
        }
        _ => unreachable!("ACL executor received an unrelated command"),
    }
}
