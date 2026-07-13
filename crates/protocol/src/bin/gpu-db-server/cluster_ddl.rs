// Legacy cluster-object DDL ownership. This is not a product execution path.

use super::{
    database_exists, tablespace_exists, write_command_complete, write_error, CatalogCommentTarget,
    Command, DatabaseInfo, ErrorField, ReadWrite, Session, TablespaceInfo,
};
use std::collections::BTreeSet;
use std::io;

pub(super) fn execute_cluster_ddl(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    command: Command,
) -> io::Result<()> {
    match command {
        Command::CreateDatabase(create) => {
            if database_exists(session, &create.name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P04",
                        message: "database already exists",
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
            session.databases.insert(
                create.name.clone(),
                DatabaseInfo {
                    oid,
                    name: create.name.clone(),
                },
            );
            session.mark_database_dirty(create.name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE DATABASE")
        }
        Command::DropDatabase(drop) => {
            let mut seen = BTreeSet::new();
            for database in &drop.names {
                if !seen.insert(database) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42710",
                            message: "database specified more than once",
                            position: None,
                        },
                    );
                }
                if database == "postgres" {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "0A000",
                            message: "cannot drop bootstrap database",
                            position: None,
                        },
                    );
                }
                if !drop.if_exists && !session.databases.contains_key(database) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "3D000",
                            message: "database does not exist",
                            position: None,
                        },
                    );
                }
            }
            for database in &drop.names {
                if session.databases.remove(database).is_some() {
                    let target = CatalogCommentTarget::Database {
                        database: database.clone(),
                    };
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                    session.database_acls.remove(database);
                    session.mark_database_acl_dirty(database.clone());
                }
                session.mark_database_dirty(database.clone());
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP DATABASE")
        }
        Command::RenameDatabase(rename) => {
            if rename.old_name == "postgres" {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "cannot rename bootstrap database",
                        position: None,
                    },
                );
            }
            if !session.databases.contains_key(&rename.old_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "3D000",
                        message: "database does not exist",
                        position: None,
                    },
                );
            }
            if database_exists(session, &rename.new_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42P04",
                        message: "database already exists",
                        position: None,
                    },
                );
            }
            let mut database = session
                .databases
                .remove(&rename.old_name)
                .expect("database existence validated");
            database.name = rename.new_name.clone();
            session.databases.insert(rename.new_name.clone(), database);
            session.mark_database_dirty(rename.old_name.clone());
            session.mark_database_dirty(rename.new_name.clone());
            if let Some(acl) = session.database_acls.remove(&rename.old_name) {
                session.database_acls.insert(rename.new_name.clone(), acl);
                session.mark_database_acl_dirty(rename.old_name.clone());
                session.mark_database_acl_dirty(rename.new_name.clone());
            }
            let old_target = CatalogCommentTarget::Database {
                database: rename.old_name,
            };
            if let Some(comment) = session.comments.remove(&old_target) {
                session.mark_comment_dirty(old_target);
                let new_target = CatalogCommentTarget::Database {
                    database: rename.new_name,
                };
                session.comments.insert(new_target.clone(), comment);
                session.mark_comment_dirty(new_target);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER DATABASE")
        }
        Command::CreateTablespace(create) => {
            if tablespace_exists(session, &create.name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42710",
                        message: "tablespace already exists",
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
            session.tablespaces.insert(
                create.name.clone(),
                TablespaceInfo {
                    oid,
                    name: create.name.clone(),
                    location: create.location,
                },
            );
            session.mark_tablespace_dirty(create.name);
            session.persist_catalog_snapshot();
            write_command_complete(stream, "CREATE TABLESPACE")
        }
        Command::DropTablespace(drop) => {
            let mut seen = BTreeSet::new();
            for tablespace in &drop.names {
                if !seen.insert(tablespace) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42710",
                            message: "tablespace specified more than once",
                            position: None,
                        },
                    );
                }
                if matches!(tablespace.as_str(), "pg_default" | "pg_global") {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "0A000",
                            message: "cannot drop bootstrap tablespace",
                            position: None,
                        },
                    );
                }
                if !drop.if_exists && !session.tablespaces.contains_key(tablespace) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42704",
                            message: "tablespace does not exist",
                            position: None,
                        },
                    );
                }
            }
            for tablespace in &drop.names {
                if session.tablespaces.remove(tablespace).is_some() {
                    let target = CatalogCommentTarget::Tablespace {
                        tablespace: tablespace.clone(),
                    };
                    session.comments.remove(&target);
                    session.mark_comment_dirty(target);
                    session.tablespace_acls.remove(tablespace);
                    session.mark_tablespace_acl_dirty(tablespace.clone());
                }
                session.mark_tablespace_dirty(tablespace.clone());
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "DROP TABLESPACE")
        }
        Command::RenameTablespace(rename) => {
            if matches!(rename.old_name.as_str(), "pg_default" | "pg_global") {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "0A000",
                        message: "cannot rename bootstrap tablespace",
                        position: None,
                    },
                );
            }
            if !session.tablespaces.contains_key(&rename.old_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42704",
                        message: "tablespace does not exist",
                        position: None,
                    },
                );
            }
            if tablespace_exists(session, &rename.new_name) {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "42710",
                        message: "tablespace already exists",
                        position: None,
                    },
                );
            }
            let mut tablespace = session
                .tablespaces
                .remove(&rename.old_name)
                .expect("tablespace existence validated");
            tablespace.name = rename.new_name.clone();
            session
                .tablespaces
                .insert(rename.new_name.clone(), tablespace);
            session.mark_tablespace_dirty(rename.old_name.clone());
            session.mark_tablespace_dirty(rename.new_name.clone());
            if let Some(acl) = session.tablespace_acls.remove(&rename.old_name) {
                session.tablespace_acls.insert(rename.new_name.clone(), acl);
                session.mark_tablespace_acl_dirty(rename.old_name.clone());
                session.mark_tablespace_acl_dirty(rename.new_name.clone());
            }
            let old_target = CatalogCommentTarget::Tablespace {
                tablespace: rename.old_name,
            };
            if let Some(comment) = session.comments.remove(&old_target) {
                session.mark_comment_dirty(old_target);
                let new_target = CatalogCommentTarget::Tablespace {
                    tablespace: rename.new_name,
                };
                session.comments.insert(new_target.clone(), comment);
                session.mark_comment_dirty(new_target);
            }
            session.persist_catalog_snapshot();
            write_command_complete(stream, "ALTER TABLESPACE")
        }
        _ => unreachable!("cluster DDL executor called with an unrelated command"),
    }
}
