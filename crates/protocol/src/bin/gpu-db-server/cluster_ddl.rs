// Legacy cluster-object DDL ownership. This is not a product execution path.

use super::{
    bool_column, database_acl_display, database_exists, int4_column, tablespace_exists,
    text_column, write_command_complete, write_error, write_single_row, CatalogCommentTarget,
    Column, Command, DatabaseInfo, ErrorField, ReadWrite, Session, TablespaceInfo,
    POSTGRES_DATABASE_OID,
};
use std::collections::BTreeSet;
use std::io;

fn pg_dump_database_metadata_query() -> &'static str {
    "select tableoid, oid, datname, datdba, pg_encoding_to_char(encoding) as encoding, datcollate, datctype, datfrozenxid, datacl, acldefault('d', datdba) as acldefault, datistemplate, datconnlimit, datminmxid, datlocprovider, daticulocale, datcollversion, daticurules, (select spcname from pg_tablespace t where t.oid = dattablespace) as tablespace, shobj_description(oid, 'pg_database') as description from pg_database where datname = current_database()"
}

fn pg_dump_database_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("datname"),
        int4_column("datdba"),
        text_column("encoding"),
        text_column("datcollate"),
        text_column("datctype"),
        text_column("datfrozenxid"),
        text_column("datacl"),
        text_column("acldefault"),
        bool_column("datistemplate"),
        int4_column("datconnlimit"),
        text_column("datminmxid"),
        text_column("datlocprovider"),
        text_column("daticulocale"),
        text_column("datcollversion"),
        text_column("daticurules"),
        text_column("tablespace"),
        text_column("description"),
    ]
}

fn pg_dump_database_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("1262".to_string()),
        Some(POSTGRES_DATABASE_OID.to_string()),
        Some("postgres".to_string()),
        Some("10".to_string()),
        Some("UTF8".to_string()),
        Some("C.UTF-8".to_string()),
        Some("C.UTF-8".to_string()),
        Some("0".to_string()),
        None,
        None,
        Some("f".to_string()),
        Some("-1".to_string()),
        Some("0".to_string()),
        Some("c".to_string()),
        None,
        None,
        None,
        Some("pg_default".to_string()),
        session
            .comments
            .get(&CatalogCommentTarget::Database {
                database: "postgres".to_string(),
            })
            .cloned(),
    ]]
}

fn psql_list_databases_catalog_query() -> &'static str {
    "select d.datname as \"name\", pg_catalog.pg_get_userbyid(d.datdba) as \"owner\", pg_catalog.pg_encoding_to_char(d.encoding) as \"encoding\", case d.datlocprovider when 'c' then 'libc' when 'i' then 'icu' end as \"locale provider\", d.datcollate as \"collate\", d.datctype as \"ctype\", d.daticulocale as \"icu locale\", d.daticurules as \"icu rules\", pg_catalog.array_to_string(d.datacl, e'\\n') as \"access privileges\" from pg_catalog.pg_database d order by 1"
}

fn psql_list_databases_verbose_catalog_query() -> &'static str {
    "select d.datname as \"name\", pg_catalog.pg_get_userbyid(d.datdba) as \"owner\", pg_catalog.pg_encoding_to_char(d.encoding) as \"encoding\", case d.datlocprovider when 'c' then 'libc' when 'i' then 'icu' end as \"locale provider\", d.datcollate as \"collate\", d.datctype as \"ctype\", d.daticulocale as \"icu locale\", d.daticurules as \"icu rules\", pg_catalog.array_to_string(d.datacl, e'\\n') as \"access privileges\", case when pg_catalog.has_database_privilege(d.datname, 'connect') then pg_catalog.pg_size_pretty(pg_catalog.pg_database_size(d.datname)) else 'no access' end as \"size\", t.spcname as \"tablespace\", pg_catalog.shobj_description(d.oid, 'pg_database') as \"description\" from pg_catalog.pg_database d join pg_catalog.pg_tablespace t on d.dattablespace = t.oid order by 1"
}

fn database_catalog_base_row(name: &str) -> Vec<Option<String>> {
    vec![
        Some(name.to_string()),
        Some("postgres".to_string()),
        Some("UTF8".to_string()),
        Some("libc".to_string()),
        Some("C.UTF-8".to_string()),
        Some("C.UTF-8".to_string()),
        None,
        None,
        None,
    ]
}

fn catalog_psql_list_database_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut names = vec!["postgres".to_string()];
    names.extend(
        session
            .databases
            .values()
            .map(|database| database.name.clone()),
    );
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let mut row = database_catalog_base_row(&name);
            row[8] = database_acl_display(session, &name);
            row
        })
        .collect()
}

fn catalog_psql_list_database_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut databases = vec![DatabaseInfo {
        oid: POSTGRES_DATABASE_OID,
        name: "postgres".to_string(),
    }];
    databases.extend(session.databases.values().cloned());
    databases.sort_by_key(|database| database.name.clone());
    databases
        .into_iter()
        .map(|database| {
            vec![
                Some(database.name.clone()),
                Some("postgres".to_string()),
                Some("UTF8".to_string()),
                Some("libc".to_string()),
                Some("C.UTF-8".to_string()),
                Some("C.UTF-8".to_string()),
                None,
                None,
                database_acl_display(session, &database.name),
                Some("0 bytes".to_string()),
                Some("pg_default".to_string()),
                session
                    .comments
                    .get(&CatalogCommentTarget::Database {
                        database: database.name.clone(),
                    })
                    .cloned(),
            ]
        })
        .collect()
}

fn catalog_database_oid_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut databases = vec![DatabaseInfo {
        oid: POSTGRES_DATABASE_OID,
        name: "postgres".to_string(),
    }];
    databases.extend(session.databases.values().cloned());
    databases.sort_by_key(|database| database.name.clone());
    databases
        .into_iter()
        .map(|database| vec![Some(database.oid.to_string()), Some(database.name)])
        .collect()
}

fn catalog_database_acl_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut names = vec!["postgres".to_string()];
    names.extend(
        session
            .databases
            .values()
            .map(|database| database.name.clone()),
    );
    names.sort();
    names
        .into_iter()
        .map(|name| vec![Some(name.clone()), database_acl_display(session, &name)])
        .collect()
}

pub(super) fn try_execute_database_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == psql_list_databases_catalog_query() {
        Some(write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Encoding"),
                text_column("Locale Provider"),
                text_column("Collate"),
                text_column("Ctype"),
                text_column("ICU Locale"),
                text_column("ICU Rules"),
                text_column("Access privileges"),
            ],
            &catalog_psql_list_database_rows(session),
        ))
    } else if canonical == psql_list_databases_verbose_catalog_query() {
        Some(write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Encoding"),
                text_column("Locale Provider"),
                text_column("Collate"),
                text_column("Ctype"),
                text_column("ICU Locale"),
                text_column("ICU Rules"),
                text_column("Access privileges"),
                text_column("Size"),
                text_column("Tablespace"),
                text_column("Description"),
            ],
            &catalog_psql_list_database_verbose_rows(session),
        ))
    } else if canonical == "select oid, datname from pg_catalog.pg_database order by datname" {
        Some(write_single_row(
            stream,
            &[int4_column("oid"), text_column("datname")],
            &catalog_database_oid_rows(session),
        ))
    } else if canonical
        == "select datname, pg_catalog.array_to_string(datacl, e'\\n') as acl from pg_catalog.pg_database order by datname"
    {
        Some(write_single_row(
            stream,
            &[text_column("datname"), text_column("acl")],
            &catalog_database_acl_rows(session),
        ))
    } else {
        None
    }
}

pub(super) fn try_execute_database_pg_dump_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical != pg_dump_database_metadata_query() {
        return None;
    }
    Some(write_single_row(
        stream,
        &pg_dump_database_metadata_columns(),
        &pg_dump_database_metadata_rows(session),
    ))
}

#[cfg(test)]
pub(super) fn test_psql_list_databases_catalog_query() -> &'static str {
    psql_list_databases_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_list_databases_verbose_catalog_query() -> &'static str {
    psql_list_databases_verbose_catalog_query()
}

#[cfg(test)]
pub(super) fn test_catalog_psql_list_database_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_list_database_rows(session)
}

#[cfg(test)]
pub(super) fn test_catalog_psql_list_database_verbose_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_list_database_verbose_rows(session)
}

#[cfg(test)]
pub(super) fn test_catalog_database_oid_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_database_oid_rows(session)
}

#[cfg(test)]
pub(super) fn test_catalog_database_acl_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_database_acl_rows(session)
}

#[cfg(test)]
pub(super) fn test_pg_dump_database_metadata_query() -> &'static str {
    pg_dump_database_metadata_query()
}

#[cfg(test)]
pub(super) fn test_pg_dump_database_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_dump_database_metadata_rows(session)
}

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
