// Legacy ACL mutation ownership. This is not a product execution path.

use super::{
    write_command_complete, write_error, AclRelationKind, CatalogCommentTarget, Command,
    DatabasePrivilege, ErrorField, FunctionPrivilege, ReadWrite, SchemaPrivilege, Session,
    TablePrivilege, TablespacePrivilege,
};
use std::io;

fn acl_relation_kind(session: &Session, relation: &str) -> Option<AclRelationKind> {
    if session.tables.contains_key(relation) {
        Some(AclRelationKind::Table)
    } else if session.views.contains_key(relation) {
        Some(AclRelationKind::View)
    } else if session.materialized_views.contains_key(relation) {
        Some(AclRelationKind::MaterializedView)
    } else if session.sequences.contains_key(relation) {
        Some(AclRelationKind::Sequence)
    } else {
        None
    }
}

pub(super) fn role_exists(session: &Session, role: &str) -> bool {
    role == "postgres" || session.roles.contains_key(role)
}

pub(super) fn database_exists(session: &Session, database: &str) -> bool {
    database == "postgres" || session.databases.contains_key(database)
}

pub(super) fn tablespace_exists(session: &Session, tablespace: &str) -> bool {
    matches!(tablespace, "pg_default" | "pg_global") || session.tablespaces.contains_key(tablespace)
}

fn acl_grantee_error(session: &Session, grantee: &str) -> Option<ErrorField> {
    if grantee == "public" || role_exists(session, grantee) {
        None
    } else {
        Some(ErrorField {
            code: "42704",
            message: "role does not exist",
            position: None,
        })
    }
}

fn active_role(session: &Session) -> &str {
    session.current_role.as_deref().unwrap_or("postgres")
}

fn role_has_relation_privilege(
    session: &Session,
    relation: &str,
    privilege: TablePrivilege,
) -> bool {
    let role = active_role(session);
    if role == "postgres" {
        return true;
    }
    session.table_acls.get(relation).is_some_and(|acl| {
        acl.get(role)
            .is_some_and(|privileges| privileges.contains(&privilege))
            || acl
                .get("public")
                .is_some_and(|privileges| privileges.contains(&privilege))
    })
}

fn role_has_function_privilege(
    session: &Session,
    function: &str,
    privilege: FunctionPrivilege,
) -> bool {
    let role = active_role(session);
    if role == "postgres" {
        return true;
    }
    session.functions.get(function).is_some_and(|function| {
        function
            .acl
            .get(role)
            .is_some_and(|privileges| privileges.contains(&privilege))
            || function
                .acl
                .get("public")
                .is_some_and(|privileges| privileges.contains(&privilege))
    })
}

fn relation_permission_error(
    session: &Session,
    relation: &str,
    privilege: TablePrivilege,
) -> Option<ErrorField> {
    if role_has_relation_privilege(session, relation, privilege) {
        None
    } else {
        Some(ErrorField {
            code: "42501",
            message: "permission denied for relation",
            position: None,
        })
    }
}

fn role_has_schema_privilege(session: &Session, schema: &str, privilege: SchemaPrivilege) -> bool {
    let role = active_role(session);
    if role == "postgres" {
        return true;
    }
    if schema != "public" || !session.public_schema_exists {
        return false;
    }
    session
        .schema_acl
        .get(role)
        .is_some_and(|privileges| privileges.contains(&privilege))
        || session
            .schema_acl
            .get("public")
            .is_some_and(|privileges| privileges.contains(&privilege))
}

fn role_has_schema_usage(session: &Session, schema: &str) -> bool {
    if active_role(session) == "postgres" {
        return true;
    }
    if schema != "public" || !session.public_schema_exists {
        return false;
    }
    if session.schema_acl.is_empty() {
        return true;
    }
    role_has_schema_privilege(session, schema, SchemaPrivilege::Usage)
}

pub(super) fn schema_permission_error(
    session: &Session,
    schema: &str,
    privilege: SchemaPrivilege,
) -> Option<ErrorField> {
    if role_has_schema_privilege(session, schema, privilege) {
        None
    } else {
        Some(ErrorField {
            code: "42501",
            message: "permission denied for schema",
            position: None,
        })
    }
}

pub(super) fn schema_usage_permission_error(session: &Session, schema: &str) -> Option<ErrorField> {
    if role_has_schema_usage(session, schema) {
        None
    } else {
        Some(ErrorField {
            code: "42501",
            message: "permission denied for schema",
            position: None,
        })
    }
}

fn function_permission_error(session: &Session, function: &str) -> Option<ErrorField> {
    if role_has_function_privilege(session, function, FunctionPrivilege::Execute) {
        None
    } else {
        Some(ErrorField {
            code: "42501",
            message: "permission denied for function",
            position: None,
        })
    }
}

pub(super) fn object_access_permission_error(
    session: &Session,
    relation: &str,
    privilege: TablePrivilege,
) -> Option<ErrorField> {
    schema_usage_permission_error(session, "public")
        .or_else(|| relation_permission_error(session, relation, privilege))
}

pub(super) fn function_access_permission_error(
    session: &Session,
    function: &str,
) -> Option<ErrorField> {
    schema_usage_permission_error(session, "public")
        .or_else(|| function_permission_error(session, function))
}

pub(super) fn role_has_dependencies(session: &Session, role: &str) -> bool {
    session.comments.contains_key(&CatalogCommentTarget::Role {
        role: role.to_string(),
    }) || session
        .table_acls
        .values()
        .any(|acl| acl.contains_key(role))
        || session
            .database_acls
            .values()
            .any(|acl| acl.contains_key(role))
        || session
            .tablespace_acls
            .values()
            .any(|acl| acl.contains_key(role))
        || session
            .functions
            .values()
            .any(|function| function.acl.contains_key(role))
        || session.schema_acl.contains_key(role)
        || session.default_table_acl.contains_key(role)
}

fn function_acl_target_error(session: &Session, function: &str) -> Option<ErrorField> {
    if session.functions.contains_key(function) {
        None
    } else {
        Some(ErrorField {
            code: "42883",
            message: "function does not exist",
            position: None,
        })
    }
}

fn relation_acl_target_error(
    session: &Session,
    relation: &str,
    kind: AclRelationKind,
) -> Option<ErrorField> {
    let Some(actual) = acl_relation_kind(session, relation) else {
        return Some(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    let table_keyword_matches_relation = kind == AclRelationKind::Table
        && matches!(
            actual,
            AclRelationKind::Table | AclRelationKind::View | AclRelationKind::MaterializedView
        );
    if kind != AclRelationKind::Relation && kind != actual && !table_keyword_matches_relation {
        return Some(ErrorField {
            code: "42809",
            message: acl_relation_kind_error(kind),
            position: None,
        });
    }
    None
}

fn acl_relation_kind_error(kind: AclRelationKind) -> &'static str {
    match kind {
        AclRelationKind::Relation => "relation does not exist",
        AclRelationKind::Table => "relation is not a table",
        AclRelationKind::View => "relation is not a view",
        AclRelationKind::MaterializedView => "relation is not a materialized view",
        AclRelationKind::Sequence => "relation is not a sequence",
    }
}

pub(super) fn grant_relation_acl(
    session: &mut Session,
    relation: &str,
    kind: AclRelationKind,
    grantee: &str,
    privileges: &[TablePrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if let Some(error) = relation_acl_target_error(session, relation, kind) {
        return Err(error);
    }
    let grantee_acl = session
        .table_acls
        .entry(relation.to_string())
        .or_default()
        .entry(grantee.to_string())
        .or_default();
    for privilege in privileges {
        grantee_acl.insert(*privilege);
    }
    session.mark_table_acl_dirty(relation.to_string());
    Ok(())
}

pub(super) fn revoke_relation_acl(
    session: &mut Session,
    relation: &str,
    kind: AclRelationKind,
    grantee: &str,
    privileges: &[TablePrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if let Some(error) = relation_acl_target_error(session, relation, kind) {
        return Err(error);
    }
    let remove_table_acl = if let Some(acl) = session.table_acls.get_mut(relation) {
        if let Some(grantee_acl) = acl.get_mut(grantee) {
            for privilege in privileges {
                grantee_acl.remove(privilege);
            }
            if grantee_acl.is_empty() {
                acl.remove(grantee);
            }
        }
        acl.is_empty()
    } else {
        false
    };
    if remove_table_acl {
        session.table_acls.remove(relation);
    }
    session.mark_table_acl_dirty(relation.to_string());
    Ok(())
}

pub(super) fn grant_function_acl(
    session: &mut Session,
    function: &str,
    grantee: &str,
    privileges: &[FunctionPrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if let Some(error) = function_acl_target_error(session, function) {
        return Err(error);
    }
    let acl = session
        .functions
        .get_mut(function)
        .expect("function ACL target preflighted")
        .acl
        .entry(grantee.to_string())
        .or_default();
    for privilege in privileges {
        acl.insert(*privilege);
    }
    session.mark_function_dirty(function.to_string());
    Ok(())
}

fn revoke_function_acl(
    session: &mut Session,
    function: &str,
    grantee: &str,
    privileges: &[FunctionPrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if let Some(error) = function_acl_target_error(session, function) {
        return Err(error);
    }
    let function_info = session
        .functions
        .get_mut(function)
        .expect("function ACL target preflighted");
    if let Some(acl) = function_info.acl.get_mut(grantee) {
        for privilege in privileges {
            acl.remove(privilege);
        }
        if acl.is_empty() {
            function_info.acl.remove(grantee);
        }
    }
    session.mark_function_dirty(function.to_string());
    Ok(())
}

fn schema_acl_target_error(session: &Session, schema: &str) -> Option<ErrorField> {
    if schema != "public" || !session.public_schema_exists {
        Some(ErrorField {
            code: "3F000",
            message: "schema does not exist",
            position: None,
        })
    } else {
        None
    }
}

pub(super) fn grant_schema_acl(
    session: &mut Session,
    schema: &str,
    grantee: &str,
    privileges: &[SchemaPrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if let Some(error) = schema_acl_target_error(session, schema) {
        return Err(error);
    }
    let acl = session.schema_acl.entry(grantee.to_string()).or_default();
    for privilege in privileges {
        acl.insert(*privilege);
    }
    session.mark_schema_acl_dirty();
    Ok(())
}

fn revoke_schema_acl(
    session: &mut Session,
    schema: &str,
    grantee: &str,
    privileges: &[SchemaPrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if let Some(error) = schema_acl_target_error(session, schema) {
        return Err(error);
    }
    if let Some(acl) = session.schema_acl.get_mut(grantee) {
        for privilege in privileges {
            acl.remove(privilege);
        }
        if acl.is_empty() {
            session.schema_acl.remove(grantee);
        }
    }
    session.mark_schema_acl_dirty();
    Ok(())
}

pub(super) fn grant_default_table_acl(
    session: &mut Session,
    grantee: &str,
    privileges: &[TablePrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    let acl = session
        .default_table_acl
        .entry(grantee.to_string())
        .or_default();
    for privilege in privileges {
        acl.insert(*privilege);
    }
    session.mark_default_table_acl_dirty();
    Ok(())
}

fn revoke_default_table_acl(
    session: &mut Session,
    grantee: &str,
    privileges: &[TablePrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if let Some(acl) = session.default_table_acl.get_mut(grantee) {
        for privilege in privileges {
            acl.remove(privilege);
        }
        if acl.is_empty() {
            session.default_table_acl.remove(grantee);
        }
    }
    session.mark_default_table_acl_dirty();
    Ok(())
}

fn grant_database_acl(
    session: &mut Session,
    database: &str,
    grantee: &str,
    privileges: &[DatabasePrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if !session.databases.contains_key(database) {
        return Err(ErrorField {
            code: "3D000",
            message: "database does not exist",
            position: None,
        });
    }
    let acl = session
        .database_acls
        .entry(database.to_string())
        .or_default()
        .entry(grantee.to_string())
        .or_default();
    for privilege in privileges {
        acl.insert(*privilege);
    }
    session.mark_database_acl_dirty(database.to_string());
    Ok(())
}

fn revoke_database_acl(
    session: &mut Session,
    database: &str,
    grantee: &str,
    privileges: &[DatabasePrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if !session.databases.contains_key(database) {
        return Err(ErrorField {
            code: "3D000",
            message: "database does not exist",
            position: None,
        });
    }
    let remove_acl = if let Some(acl) = session.database_acls.get_mut(database) {
        if let Some(grantee_acl) = acl.get_mut(grantee) {
            for privilege in privileges {
                grantee_acl.remove(privilege);
            }
            if grantee_acl.is_empty() {
                acl.remove(grantee);
            }
        }
        acl.is_empty()
    } else {
        false
    };
    if remove_acl {
        session.database_acls.remove(database);
    }
    session.mark_database_acl_dirty(database.to_string());
    Ok(())
}

fn grant_tablespace_acl(
    session: &mut Session,
    tablespace: &str,
    grantee: &str,
    privileges: &[TablespacePrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if !session.tablespaces.contains_key(tablespace) {
        return Err(ErrorField {
            code: "42704",
            message: "tablespace does not exist",
            position: None,
        });
    }
    let acl = session
        .tablespace_acls
        .entry(tablespace.to_string())
        .or_default()
        .entry(grantee.to_string())
        .or_default();
    for privilege in privileges {
        acl.insert(*privilege);
    }
    session.mark_tablespace_acl_dirty(tablespace.to_string());
    Ok(())
}

fn revoke_tablespace_acl(
    session: &mut Session,
    tablespace: &str,
    grantee: &str,
    privileges: &[TablespacePrivilege],
) -> Result<(), ErrorField> {
    if let Some(error) = acl_grantee_error(session, grantee) {
        return Err(error);
    }
    if !session.tablespaces.contains_key(tablespace) {
        return Err(ErrorField {
            code: "42704",
            message: "tablespace does not exist",
            position: None,
        });
    }
    let remove_acl = if let Some(acl) = session.tablespace_acls.get_mut(tablespace) {
        if let Some(grantee_acl) = acl.get_mut(grantee) {
            for privilege in privileges {
                grantee_acl.remove(privilege);
            }
            if grantee_acl.is_empty() {
                acl.remove(grantee);
            }
        }
        acl.is_empty()
    } else {
        false
    };
    if remove_acl {
        session.tablespace_acls.remove(tablespace);
    }
    session.mark_tablespace_acl_dirty(tablespace.to_string());
    Ok(())
}

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
