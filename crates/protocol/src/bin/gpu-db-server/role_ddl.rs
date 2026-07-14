// Legacy role DDL ownership. This is not a product execution path.

use super::{
    bool_column, database_acl_display, int4_column, role_exists, role_has_dependencies,
    text_column, write_command_complete, write_error, write_single_row, CatalogCommentTarget,
    Column, Command, ErrorField, ReadWrite, RoleInfo, Session,
};
use std::collections::BTreeSet;
use std::io;

fn psql_describe_roles_catalog_query() -> &'static str {
    "select r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, r.rolcreatedb, r.rolcanlogin, r.rolconnlimit, r.rolvaliduntil , r.rolreplication , r.rolbypassrls from pg_catalog.pg_roles r where r.rolname !~ '^pg_' order by 1"
}

fn psql_describe_roles_verbose_catalog_query() -> &'static str {
    "select r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, r.rolcreatedb, r.rolcanlogin, r.rolconnlimit, r.rolvaliduntil , pg_catalog.shobj_description(r.oid, 'pg_authid') as description , r.rolreplication , r.rolbypassrls from pg_catalog.pg_roles r where r.rolname !~ '^pg_' order by 1"
}

fn pg_dumpall_role_metadata_query() -> &'static str {
    "select oid, rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolcanlogin, rolconnlimit, rolpassword, rolvaliduntil, rolreplication, rolbypassrls, pg_catalog.shobj_description(oid, 'pg_authid') as rolcomment, rolname = current_user as is_current_user from pg_roles where rolname !~ '^pg_' order by 2"
}

fn pg_dumpall_role_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("oid"),
        text_column("rolname"),
        bool_column("rolsuper"),
        bool_column("rolinherit"),
        bool_column("rolcreaterole"),
        bool_column("rolcreatedb"),
        bool_column("rolcanlogin"),
        int4_column("rolconnlimit"),
        text_column("rolpassword"),
        text_column("rolvaliduntil"),
        bool_column("rolreplication"),
        bool_column("rolbypassrls"),
        text_column("rolcomment"),
        bool_column("is_current_user"),
    ]
}

fn pg_dumpall_role_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = vec![vec![
        Some("10".to_string()),
        Some("postgres".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("-1".to_string()),
        None,
        None,
        Some("t".to_string()),
        Some("t".to_string()),
        session
            .comments
            .get(&CatalogCommentTarget::Role {
                role: "postgres".to_string(),
            })
            .cloned(),
        Some("t".to_string()),
    ]];
    let mut roles = session.roles.values().collect::<Vec<_>>();
    roles.sort_by(|left, right| left.name.cmp(&right.name));
    rows.extend(roles.into_iter().map(|role| {
        vec![
            Some(role.oid.to_string()),
            Some(role.name.clone()),
            Some("f".to_string()),
            Some("t".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some(if role.login { "t" } else { "f" }.to_string()),
            Some("-1".to_string()),
            None,
            None,
            Some("f".to_string()),
            Some("f".to_string()),
            session
                .comments
                .get(&CatalogCommentTarget::Role {
                    role: role.name.clone(),
                })
                .cloned(),
            Some("f".to_string()),
        ]
    }));
    rows
}

fn catalog_psql_describe_role_rows(session: &Session, verbose: bool) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut row = vec![
        Some("postgres".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("-1".to_string()),
        database_acl_display(session, "postgres"),
    ];
    if verbose {
        row.push(
            session
                .comments
                .get(&CatalogCommentTarget::Role {
                    role: "postgres".to_string(),
                })
                .cloned(),
        );
    }
    row.extend([Some("t".to_string()), Some("t".to_string())]);
    rows.push(row);
    for role in session.roles.values() {
        let mut row = vec![
            Some(role.name.clone()),
            Some("f".to_string()),
            Some("t".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some(if role.login { "t" } else { "f" }.to_string()),
            Some("-1".to_string()),
            None,
        ];
        if verbose {
            row.push(
                session
                    .comments
                    .get(&CatalogCommentTarget::Role {
                        role: role.name.clone(),
                    })
                    .cloned(),
            );
        }
        row.extend([Some("f".to_string()), Some("f".to_string())]);
        rows.push(row);
    }
    rows
}

fn catalog_role_oid_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = vec![vec![Some("10".to_string()), Some("postgres".to_string())]];
    rows.extend(
        session
            .roles
            .values()
            .map(|role| vec![Some(role.oid.to_string()), Some(role.name.clone())]),
    );
    rows
}

pub(super) fn try_execute_role_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical != psql_describe_roles_catalog_query()
        && canonical != psql_describe_roles_verbose_catalog_query()
    {
        return None;
    }
    let verbose = canonical == psql_describe_roles_verbose_catalog_query();
    let mut columns = vec![
        text_column("rolname"),
        bool_column("rolsuper"),
        bool_column("rolinherit"),
        bool_column("rolcreaterole"),
        bool_column("rolcreatedb"),
        bool_column("rolcanlogin"),
        int4_column("rolconnlimit"),
        text_column("rolvaliduntil"),
    ];
    if verbose {
        columns.push(text_column("Description"));
    }
    columns.push(bool_column("rolreplication"));
    columns.push(bool_column("rolbypassrls"));
    Some(write_single_row(
        stream,
        &columns,
        &catalog_psql_describe_role_rows(session, verbose),
    ))
}

pub(super) fn try_execute_role_pg_dump_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical == "select oid, rolname from pg_catalog.pg_roles order by 1" {
        Some(write_single_row(
            stream,
            &[int4_column("oid"), text_column("rolname")],
            &catalog_role_oid_rows(session),
        ))
    } else if canonical == pg_dumpall_role_metadata_query() {
        Some(write_single_row(
            stream,
            &pg_dumpall_role_metadata_columns(),
            &pg_dumpall_role_metadata_rows(session),
        ))
    } else {
        None
    }
}

#[cfg(test)]
pub(super) fn test_psql_describe_roles_catalog_query() -> &'static str {
    psql_describe_roles_catalog_query()
}

#[cfg(test)]
pub(super) fn test_psql_describe_roles_verbose_catalog_query() -> &'static str {
    psql_describe_roles_verbose_catalog_query()
}

#[cfg(test)]
pub(super) fn test_catalog_psql_describe_role_rows(
    session: &Session,
    verbose: bool,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_role_rows(session, verbose)
}

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
