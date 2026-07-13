// Legacy role DDL ownership. This is not a product execution path.

use super::{
    role_exists, role_has_dependencies, write_command_complete, write_error, CatalogCommentTarget,
    Command, ErrorField, ReadWrite, RoleInfo, Session,
};
use std::collections::BTreeSet;
use std::io;

pub(super) fn execute_role_ddl(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CreateRole(create) => {
            if role_exists(session, &create.name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42710",
                        message: "role already exists",
                        position: None,
                    },
                );
            }
            let oid = session.next_relation_oid;
            let Some(next_oid) = session.next_relation_oid.checked_add(1) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "54000",
                        message: "relational OID counter overflow",
                        position: None,
                    },
                );
            };
            session.next_relation_oid = next_oid;
            session.roles.insert(
                create.name.clone(),
                RoleInfo {
                    oid,
                    name: create.name.clone(),
                    login: create.login,
                },
            );
            session.mark_role_dirty(create.name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE ROLE")
        }
        Command::DropRole(drop) => {
            let mut seen = BTreeSet::new();
            for role in &drop.names {
                if !seen.insert(role) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42710",
                            message: "role specified more than once",
                            position: None,
                        },
                    );
                }
                if role == "postgres" {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "0A000",
                            message: "cannot drop bootstrap role",
                            position: None,
                        },
                    );
                }
                if !drop.if_exists && !session.roles.contains_key(role) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42704",
                            message: "role does not exist",
                            position: None,
                        },
                    );
                }
                if session.roles.contains_key(role) && role_has_dependencies(session, role) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "2BP01",
                            message: "role cannot be dropped because dependent metadata exists",
                            position: None,
                        },
                    );
                }
            }
            for role in drop.names {
                session.roles.remove(&role);
                session.mark_role_dirty(role);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP ROLE")
        }
        Command::RenameRole(rename) => {
            if rename.old_name == "postgres" {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "cannot rename bootstrap role",
                        position: None,
                    },
                );
            }
            if !session.roles.contains_key(&rename.old_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42704",
                        message: "role does not exist",
                        position: None,
                    },
                );
            }
            if role_exists(session, &rename.new_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42710",
                        message: "role already exists",
                        position: None,
                    },
                );
            }
            let mut role = session
                .roles
                .remove(&rename.old_name)
                .expect("role existence validated");
            role.name = rename.new_name.clone();
            session.roles.insert(rename.new_name.clone(), role);
            session.mark_role_dirty(rename.old_name.clone());
            session.mark_role_dirty(rename.new_name.clone());
            let old_target = CatalogCommentTarget::Role {
                role: rename.old_name.clone(),
            };
            if let Some(comment) = session.comments.remove(&old_target) {
                session.mark_comment_dirty(old_target);
                let new_target = CatalogCommentTarget::Role {
                    role: rename.new_name.clone(),
                };
                session.comments.insert(new_target.clone(), comment);
                session.mark_comment_dirty(new_target);
            }
            for acl in session.table_acls.values_mut() {
                if let Some(privileges) = acl.remove(&rename.old_name) {
                    acl.insert(rename.new_name.clone(), privileges);
                }
            }
            for table in session.table_acls.keys().cloned().collect::<Vec<_>>() {
                session.mark_table_acl_dirty(table);
            }
            for acl in session.database_acls.values_mut() {
                if let Some(privileges) = acl.remove(&rename.old_name) {
                    acl.insert(rename.new_name.clone(), privileges);
                }
            }
            for database in session.database_acls.keys().cloned().collect::<Vec<_>>() {
                session.mark_database_acl_dirty(database);
            }
            for acl in session.tablespace_acls.values_mut() {
                if let Some(privileges) = acl.remove(&rename.old_name) {
                    acl.insert(rename.new_name.clone(), privileges);
                }
            }
            for tablespace in session.tablespace_acls.keys().cloned().collect::<Vec<_>>() {
                session.mark_tablespace_acl_dirty(tablespace);
            }
            let mut dirty_functions = Vec::new();
            for function in session.functions.values_mut() {
                if let Some(privileges) = function.acl.remove(&rename.old_name) {
                    function.acl.insert(rename.new_name.clone(), privileges);
                    dirty_functions.push(function.name.clone());
                }
            }
            for function in dirty_functions {
                session.mark_function_dirty(function);
            }
            if let Some(privileges) = session.schema_acl.remove(&rename.old_name) {
                session
                    .schema_acl
                    .insert(rename.new_name.clone(), privileges);
                session.mark_schema_acl_dirty();
            }
            if let Some(privileges) = session.default_table_acl.remove(&rename.old_name) {
                session
                    .default_table_acl
                    .insert(rename.new_name.clone(), privileges);
                session.mark_default_table_acl_dirty();
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER ROLE")
        }
        _ => unreachable!("role DDL executor called with an unrelated command"),
    }
}
