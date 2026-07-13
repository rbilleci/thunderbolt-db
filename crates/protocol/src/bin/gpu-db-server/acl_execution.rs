// Legacy ACL mutation ownership. This is not a product execution path.

use super::{
    int4_column, text_column, write_command_complete, write_error, write_single_row,
    AclRelationKind, CatalogCommentTarget, Column, Command, DatabasePrivilege, ErrorField,
    FunctionPrivilege, ReadWrite, SchemaPrivilege, Session, TablePrivilege, TablespacePrivilege,
    PUBLIC_NAMESPACE_OID,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io;

pub(super) fn relation_acl_array_display(session: &Session, relation: &str) -> Option<String> {
    let acl = session.table_acls.get(relation)?;
    let default = if session.sequences.contains_key(relation) {
        "postgres=rwU/postgres"
    } else {
        "postgres=arwdDxt/postgres"
    };
    acl_array_display_with_default(acl, default)
}

pub(super) fn schema_acl_display(session: &Session) -> Option<String> {
    let rows = session
        .schema_acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!(
                "{grantee}={}/postgres",
                schema_privilege_letters(privileges)
            ))
        })
        .collect::<Vec<_>>();
    (!rows.is_empty()).then(|| rows.join("\n"))
}

pub(super) fn schema_acl_array_display(session: &Session) -> Option<String> {
    let rows = session
        .schema_acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!(
                "{grantee}={}/postgres",
                schema_privilege_letters(privileges)
            ))
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        None
    } else {
        let mut with_defaults = vec![
            "postgres=UC/postgres".to_string(),
            "=U/postgres".to_string(),
        ];
        with_defaults.extend(rows);
        Some(format!("{{{}}}", with_defaults.join(",")))
    }
}

pub(super) fn database_acl_display(session: &Session, database: &str) -> Option<String> {
    let acl = session.database_acls.get(database)?;
    let rows = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!(
                "{grantee}={}/postgres",
                database_privilege_letters(privileges)
            ))
        })
        .collect::<Vec<_>>();
    (!rows.is_empty()).then(|| rows.join("\n"))
}

fn database_privilege_letters(privileges: &BTreeSet<DatabasePrivilege>) -> String {
    let mut letters = String::new();
    for (privilege, letter) in [
        (DatabasePrivilege::Connect, 'c'),
        (DatabasePrivilege::Temporary, 'T'),
    ] {
        if privileges.contains(&privilege) {
            letters.push(letter);
        }
    }
    letters
}

pub(super) fn tablespace_acl_display(session: &Session, tablespace: &str) -> Option<String> {
    let acl = session.tablespace_acls.get(tablespace)?;
    let rows = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!(
                "{grantee}={}/postgres",
                tablespace_privilege_letters(privileges)
            ))
        })
        .collect::<Vec<_>>();
    (!rows.is_empty()).then(|| rows.join("\n"))
}

pub(super) fn tablespace_acl_array_display(session: &Session, tablespace: &str) -> Option<String> {
    let acl = session.tablespace_acls.get(tablespace)?;
    let rows = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!(
                "{grantee}={}/postgres",
                tablespace_privilege_letters(privileges)
            ))
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        None
    } else {
        let mut with_default = vec!["postgres=C/postgres".to_string()];
        with_default.extend(rows);
        Some(format!("{{{}}}", with_default.join(",")))
    }
}

fn tablespace_privilege_letters(privileges: &BTreeSet<TablespacePrivilege>) -> String {
    let mut letters = String::new();
    if privileges.contains(&TablespacePrivilege::Create) {
        letters.push('C');
    }
    letters
}

pub(super) fn function_acl_array_display(
    acl: &BTreeMap<String, BTreeSet<FunctionPrivilege>>,
) -> Option<String> {
    let rows = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!(
                "{grantee}={}/postgres",
                function_privilege_letters(privileges)
            ))
        })
        .collect::<Vec<_>>();
    (!rows.is_empty()).then(|| format!("{{{}}}", rows.join(",")))
}

pub(super) fn function_privilege_letters(privileges: &BTreeSet<FunctionPrivilege>) -> String {
    let mut letters = String::new();
    if privileges.contains(&FunctionPrivilege::Execute) {
        letters.push('X');
    }
    letters
}

fn schema_privilege_letters(privileges: &BTreeSet<SchemaPrivilege>) -> String {
    let mut letters = String::new();
    for (privilege, letter) in [
        (SchemaPrivilege::Usage, 'U'),
        (SchemaPrivilege::Create, 'C'),
    ] {
        if privileges.contains(&privilege) {
            letters.push(letter);
        }
    }
    letters
}

pub(super) fn acl_display(acl: &BTreeMap<String, BTreeSet<TablePrivilege>>) -> Option<String> {
    let rows = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!(
                "{grantee}={}/postgres",
                table_privilege_letters(privileges)
            ))
        })
        .collect::<Vec<_>>();
    (!rows.is_empty()).then(|| rows.join("\n"))
}

fn acl_array_display(acl: &BTreeMap<String, BTreeSet<TablePrivilege>>) -> Option<String> {
    acl_array_display_with_default(acl, "")
}

fn acl_array_display_with_default(
    acl: &BTreeMap<String, BTreeSet<TablePrivilege>>,
    default: &str,
) -> Option<String> {
    let rows = acl
        .iter()
        .filter_map(|(grantee, privileges)| {
            if privileges.is_empty() {
                return None;
            }
            let grantee = if grantee == "public" { "" } else { grantee };
            Some(format!(
                "{grantee}={}/postgres",
                table_privilege_letters(privileges)
            ))
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return None;
    }
    let mut all_rows = Vec::new();
    if !default.is_empty() {
        all_rows.push(default.to_string());
    }
    all_rows.extend(rows);
    Some(format!("{{{}}}", all_rows.join(",")))
}

fn table_privilege_letters(privileges: &BTreeSet<TablePrivilege>) -> String {
    let mut letters = String::new();
    for (privilege, letter) in [
        (TablePrivilege::Insert, 'a'),
        (TablePrivilege::Select, 'r'),
        (TablePrivilege::Update, 'w'),
        (TablePrivilege::Delete, 'd'),
    ] {
        if privileges.contains(&privilege) {
            letters.push(letter);
        }
    }
    letters
}

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

pub(super) fn try_execute_default_acl_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical != psql_list_default_access_privileges_catalog_query() {
        return None;
    }
    Some(write_single_row(
        stream,
        &[
            text_column("Owner"),
            text_column("Schema"),
            text_column("Type"),
            text_column("Access privileges"),
        ],
        &catalog_psql_default_access_privilege_rows(session),
    ))
}

pub(super) fn try_execute_default_acl_pg_dump_catalog_query(
    stream: &mut dyn ReadWrite,
    session: &Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if !is_pg_dump_default_acl_metadata_query(canonical) {
        return None;
    }
    Some(write_single_row(
        stream,
        &pg_dump_default_acl_metadata_columns(),
        &pg_dump_default_acl_metadata_rows(session),
    ))
}

fn psql_list_default_access_privileges_catalog_query() -> &'static str {
    "select pg_catalog.pg_get_userbyid(d.defaclrole) as \"owner\", n.nspname as \"schema\", case d.defaclobjtype when 'r' then 'table' when 's' then 'sequence' when 'f' then 'function' when 't' then 'type' when 'n' then 'schema' end as \"type\", pg_catalog.array_to_string(d.defaclacl, e'\\n') as \"access privileges\" from pg_catalog.pg_default_acl d left join pg_catalog.pg_namespace n on n.oid = d.defaclnamespace order by 1, 2, 3"
}

fn catalog_psql_default_access_privilege_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    acl_display(&session.default_table_acl)
        .map(|acl| {
            vec![vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("table".to_string()),
                Some(acl),
            ]]
        })
        .unwrap_or_default()
}

fn pg_dump_default_table_acl_array_display(session: &Session) -> Option<String> {
    let acl = acl_array_display(&session.default_table_acl)?;
    let inner = acl.strip_prefix('{')?.strip_suffix('}')?;
    Some(format!("{{postgres=arwdDxt/postgres,{inner}}}"))
}

fn is_pg_dump_default_acl_metadata_query(canonical: &str) -> bool {
    canonical.starts_with("select oid, tableoid, defaclrole")
        && canonical.contains("from pg_default_acl")
}

fn pg_dump_default_acl_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("oid"),
        int4_column("tableoid"),
        int4_column("defaclrole"),
        int4_column("defaclnamespace"),
        text_column("defaclobjtype"),
        text_column("defaclacl"),
        text_column("acldefault"),
    ]
}

fn pg_dump_default_acl_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let Some(acl) = pg_dump_default_table_acl_array_display(session) else {
        return Vec::new();
    };
    vec![vec![
        Some("82600".to_string()),
        Some("826".to_string()),
        Some("10".to_string()),
        Some(PUBLIC_NAMESPACE_OID.to_string()),
        Some("r".to_string()),
        Some(acl),
        Some("{postgres=arwdDxt/postgres}".to_string()),
    ]]
}

#[cfg(test)]
pub(super) fn test_psql_list_default_access_privileges_catalog_query() -> &'static str {
    psql_list_default_access_privileges_catalog_query()
}

#[cfg(test)]
pub(super) fn test_catalog_psql_default_access_privilege_rows(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_default_access_privilege_rows(session)
}
