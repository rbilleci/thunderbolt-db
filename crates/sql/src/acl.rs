//! SQL access-control contracts and privilege parsing.

use super::{
    find_char_outside_quotes, find_keyword_outside_quotes, normalize_identifier,
    normalize_relation_identifier, split_csv, split_leading_identifier,
    strip_keyword_prefix_case_insensitive, Command, ParseError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AclRelationKind {
    Relation,
    Table,
    View,
    MaterializedView,
    Sequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TablePrivilege {
    Select,
    Insert,
    Update,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SchemaPrivilege {
    Usage,
    Create,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DatabasePrivilege {
    Connect,
    Temporary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TablespacePrivilege {
    Create,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FunctionPrivilege {
    Execute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantTable {
    pub relation: String,
    pub kind: AclRelationKind,
    pub grantee: String,
    pub privileges: Vec<TablePrivilege>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevokeTable {
    pub relation: String,
    pub kind: AclRelationKind,
    pub grantee: String,
    pub privileges: Vec<TablePrivilege>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaPrivileges {
    pub schema: String,
    pub grantee: String,
    pub privileges: Vec<SchemaPrivilege>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatabasePrivileges {
    pub database: String,
    pub grantee: String,
    pub privileges: Vec<DatabasePrivilege>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TablespacePrivileges {
    pub tablespace: String,
    pub grantee: String,
    pub privileges: Vec<TablespacePrivilege>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionPrivileges {
    pub function: String,
    pub grantee: String,
    pub privileges: Vec<FunctionPrivilege>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultTablePrivileges {
    pub grantee: String,
    pub privileges: Vec<TablePrivilege>,
}

pub(super) fn parse_acl_command(input: &str) -> Option<Result<Command, ParseError>> {
    let first = input.split_whitespace().next()?;
    if first.eq_ignore_ascii_case("ALTER")
        && strip_keyword_prefix_case_insensitive(input, "ALTER DEFAULT PRIVILEGES").is_some()
    {
        return Some(parse_alter_default_table_privileges(input));
    }
    if first.eq_ignore_ascii_case("GRANT") {
        if parse_grant_schema(input).is_ok() {
            return Some(parse_grant_schema(input).map(Command::GrantSchema));
        }
        if parse_grant_database(input).is_ok() {
            return Some(parse_grant_database(input).map(Command::GrantDatabase));
        }
        if parse_grant_tablespace(input).is_ok() {
            return Some(parse_grant_tablespace(input).map(Command::GrantTablespace));
        }
        if parse_grant_function(input).is_ok() {
            return Some(parse_grant_function(input).map(Command::GrantFunction));
        }
        return Some(parse_grant_table(input).map(Command::GrantTable));
    }
    if first.eq_ignore_ascii_case("REVOKE") {
        if parse_revoke_schema(input).is_ok() {
            return Some(parse_revoke_schema(input).map(Command::RevokeSchema));
        }
        if parse_revoke_database(input).is_ok() {
            return Some(parse_revoke_database(input).map(Command::RevokeDatabase));
        }
        if parse_revoke_tablespace(input).is_ok() {
            return Some(parse_revoke_tablespace(input).map(Command::RevokeTablespace));
        }
        if parse_revoke_function(input).is_ok() {
            return Some(parse_revoke_function(input).map(Command::RevokeFunction));
        }
        return Some(parse_revoke_table(input).map(Command::RevokeTable));
    }
    None
}

fn parse_grant_table(input: &str) -> Result<GrantTable, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "GRANT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if find_keyword_outside_quotes(rest, "WITH").is_some()
        || find_keyword_outside_quotes(rest, "GRANT OPTION").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let on_idx = find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
    let (privileges, target_and_grantee) = rest.split_at(on_idx);
    let target_and_grantee = strip_keyword_prefix_case_insensitive(target_and_grantee, "ON")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let to_idx = find_keyword_outside_quotes(target_and_grantee, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?;
    let (target, grantee) = target_and_grantee.split_at(to_idx);
    let grantee = strip_keyword_prefix_case_insensitive(grantee, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    let (relation, kind) = parse_acl_relation_target(target)?;
    Ok(GrantTable {
        relation,
        kind,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_table_privileges(privileges)?,
    })
}

fn parse_revoke_table(input: &str) -> Result<RevokeTable, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "REVOKE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "GRANT OPTION FOR").is_some()
        || find_keyword_outside_quotes(rest, "GRANT OPTION").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let on_idx = find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
    let (privileges, target_and_grantee) = rest.split_at(on_idx);
    let target_and_grantee = strip_keyword_prefix_case_insensitive(target_and_grantee, "ON")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let from_idx = find_keyword_outside_quotes(target_and_grantee, "FROM")
        .ok_or(ParseError::InvalidRelationalSql)?;
    let (target, grantee) = target_and_grantee.split_at(from_idx);
    let grantee = strip_keyword_prefix_case_insensitive(grantee, "FROM")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    let (relation, kind) = parse_acl_relation_target(target)?;
    Ok(RevokeTable {
        relation,
        kind,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_table_privileges(privileges)?,
    })
}

fn parse_grant_schema(input: &str) -> Result<SchemaPrivileges, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "GRANT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if find_keyword_outside_quotes(rest, "WITH").is_some()
        || find_keyword_outside_quotes(rest, "GRANT OPTION").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let on_idx = find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
    let (privileges, target_and_grantee) = rest.split_at(on_idx);
    let target_and_grantee = strip_keyword_prefix_case_insensitive(target_and_grantee, "ON")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let to_idx = find_keyword_outside_quotes(target_and_grantee, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?;
    let (target, grantee) = target_and_grantee.split_at(to_idx);
    let schema = parse_acl_schema_target(target)?;
    let grantee = strip_keyword_prefix_case_insensitive(grantee, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    Ok(SchemaPrivileges {
        schema,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_schema_privileges(privileges)?,
    })
}

fn parse_revoke_schema(input: &str) -> Result<SchemaPrivileges, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "REVOKE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "GRANT OPTION FOR").is_some()
        || find_keyword_outside_quotes(rest, "GRANT OPTION").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let on_idx = find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
    let (privileges, target_and_grantee) = rest.split_at(on_idx);
    let target_and_grantee = strip_keyword_prefix_case_insensitive(target_and_grantee, "ON")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let from_idx = find_keyword_outside_quotes(target_and_grantee, "FROM")
        .ok_or(ParseError::InvalidRelationalSql)?;
    let (target, grantee) = target_and_grantee.split_at(from_idx);
    let schema = parse_acl_schema_target(target)?;
    let grantee = strip_keyword_prefix_case_insensitive(grantee, "FROM")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    Ok(SchemaPrivileges {
        schema,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_schema_privileges(privileges)?,
    })
}

fn parse_grant_database(input: &str) -> Result<DatabasePrivileges, ParseError> {
    let (privileges, target, grantee) = parse_grant_acl_parts(input)?;
    let database = parse_named_acl_target(target, "DATABASE")?;
    Ok(DatabasePrivileges {
        database,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_database_privileges(privileges)?,
    })
}

fn parse_revoke_database(input: &str) -> Result<DatabasePrivileges, ParseError> {
    let (privileges, target, grantee) = parse_revoke_acl_parts(input)?;
    let database = parse_named_acl_target(target, "DATABASE")?;
    Ok(DatabasePrivileges {
        database,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_database_privileges(privileges)?,
    })
}

fn parse_grant_tablespace(input: &str) -> Result<TablespacePrivileges, ParseError> {
    let (privileges, target, grantee) = parse_grant_acl_parts(input)?;
    let tablespace = parse_named_acl_target(target, "TABLESPACE")?;
    Ok(TablespacePrivileges {
        tablespace,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_tablespace_privileges(privileges)?,
    })
}

fn parse_revoke_tablespace(input: &str) -> Result<TablespacePrivileges, ParseError> {
    let (privileges, target, grantee) = parse_revoke_acl_parts(input)?;
    let tablespace = parse_named_acl_target(target, "TABLESPACE")?;
    Ok(TablespacePrivileges {
        tablespace,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_tablespace_privileges(privileges)?,
    })
}

fn parse_grant_function(input: &str) -> Result<FunctionPrivileges, ParseError> {
    let (privileges, target, grantee) = parse_grant_acl_parts(input)?;
    let function = parse_function_acl_target(target)?;
    Ok(FunctionPrivileges {
        function,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_function_privileges(privileges)?,
    })
}

fn parse_revoke_function(input: &str) -> Result<FunctionPrivileges, ParseError> {
    let (privileges, target, grantee) = parse_revoke_acl_parts(input)?;
    let function = parse_function_acl_target(target)?;
    Ok(FunctionPrivileges {
        function,
        grantee: parse_acl_grantee(grantee)?,
        privileges: parse_function_privileges(privileges)?,
    })
}

fn parse_grant_acl_parts(input: &str) -> Result<(&str, &str, &str), ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "GRANT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if find_keyword_outside_quotes(rest, "WITH").is_some()
        || find_keyword_outside_quotes(rest, "GRANT OPTION").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let on_idx = find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
    let (privileges, target_and_grantee) = rest.split_at(on_idx);
    let target_and_grantee = strip_keyword_prefix_case_insensitive(target_and_grantee, "ON")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let to_idx = find_keyword_outside_quotes(target_and_grantee, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?;
    let (target, grantee) = target_and_grantee.split_at(to_idx);
    let grantee = strip_keyword_prefix_case_insensitive(grantee, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    Ok((privileges, target, grantee))
}

fn parse_revoke_acl_parts(input: &str) -> Result<(&str, &str, &str), ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "REVOKE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "GRANT OPTION FOR").is_some()
        || find_keyword_outside_quotes(rest, "GRANT OPTION").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let on_idx = find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
    let (privileges, target_and_grantee) = rest.split_at(on_idx);
    let target_and_grantee = strip_keyword_prefix_case_insensitive(target_and_grantee, "ON")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let from_idx = find_keyword_outside_quotes(target_and_grantee, "FROM")
        .ok_or(ParseError::InvalidRelationalSql)?;
    let (target, grantee) = target_and_grantee.split_at(from_idx);
    let grantee = strip_keyword_prefix_case_insensitive(grantee, "FROM")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    Ok((privileges, target, grantee))
}

fn parse_named_acl_target(target: &str, keyword: &str) -> Result<String, ParseError> {
    let name = strip_keyword_prefix_case_insensitive(target.trim(), keyword)
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if name.is_empty()
        || name.contains(',')
        || find_keyword_outside_quotes(name, "TABLE").is_some()
        || find_keyword_outside_quotes(name, "SCHEMA").is_some()
        || find_keyword_outside_quotes(name, "DATABASE").is_some()
        || find_keyword_outside_quotes(name, "TABLESPACE").is_some()
        || find_keyword_outside_quotes(name, "FUNCTION").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    normalize_identifier(name)
}

fn parse_function_acl_target(target: &str) -> Result<String, ParseError> {
    let mut name = strip_keyword_prefix_case_insensitive(target.trim(), "FUNCTION")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if name.is_empty()
        || name.contains(',')
        || find_keyword_outside_quotes(name, "TABLE").is_some()
        || find_keyword_outside_quotes(name, "SCHEMA").is_some()
        || find_keyword_outside_quotes(name, "SEQUENCE").is_some()
        || find_keyword_outside_quotes(name, "VIEW").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    if let Some(open_idx) = find_char_outside_quotes(name, '(') {
        let (before_args, args) = name.split_at(open_idx);
        if !args.trim().eq("()") {
            return Err(ParseError::InvalidRelationalSql);
        }
        name = before_args.trim_end();
    }
    normalize_relation_identifier(name)
}

fn parse_acl_schema_target(target: &str) -> Result<String, ParseError> {
    let schema = strip_keyword_prefix_case_insensitive(target.trim(), "SCHEMA")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if schema.is_empty()
        || schema.contains(',')
        || find_keyword_outside_quotes(schema, "TABLE").is_some()
        || find_keyword_outside_quotes(schema, "SEQUENCE").is_some()
        || find_keyword_outside_quotes(schema, "VIEW").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let schema = normalize_identifier(schema)?;
    if schema == "public" {
        Ok(schema)
    } else {
        Err(ParseError::InvalidRelationalSql)
    }
}

fn parse_alter_default_table_privileges(input: &str) -> Result<Command, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER DEFAULT PRIVILEGES")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();

    if let Some(after_for_role) = strip_keyword_prefix_case_insensitive(rest, "FOR ROLE") {
        let (role, after_role) = split_leading_identifier(after_for_role.trim_start())?;
        if normalize_identifier(role)? != "postgres" {
            return Err(ParseError::InvalidRelationalSql);
        }
        rest = after_role.trim_start();
    }
    if let Some(after_in_schema) = strip_keyword_prefix_case_insensitive(rest, "IN SCHEMA") {
        let (schema, after_schema) = split_leading_identifier(after_in_schema.trim_start())?;
        if normalize_identifier(schema)? != "public" {
            return Err(ParseError::InvalidRelationalSql);
        }
        rest = after_schema.trim_start();
    }

    if let Some(after_grant) = strip_keyword_prefix_case_insensitive(rest, "GRANT") {
        if find_keyword_outside_quotes(after_grant, "WITH").is_some()
            || find_keyword_outside_quotes(after_grant, "GRANT OPTION").is_some()
        {
            return Err(ParseError::InvalidRelationalSql);
        }
        let on_idx = find_keyword_outside_quotes(after_grant, "ON")
            .ok_or(ParseError::InvalidRelationalSql)?;
        let (privileges, target_and_grantee) = after_grant.split_at(on_idx);
        let target_and_grantee = strip_keyword_prefix_case_insensitive(target_and_grantee, "ON")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim_start();
        let to_idx = find_keyword_outside_quotes(target_and_grantee, "TO")
            .ok_or(ParseError::InvalidRelationalSql)?;
        let (target, grantee) = target_and_grantee.split_at(to_idx);
        if !target.trim().eq_ignore_ascii_case("TABLES") {
            return Err(ParseError::InvalidRelationalSql);
        }
        let grantee = strip_keyword_prefix_case_insensitive(grantee, "TO")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim();
        return Ok(Command::GrantDefaultTablePrivileges(
            DefaultTablePrivileges {
                grantee: parse_acl_grantee(grantee)?,
                privileges: parse_table_privileges(privileges)?,
            },
        ));
    }

    if let Some(after_revoke) = strip_keyword_prefix_case_insensitive(rest, "REVOKE") {
        if strip_keyword_prefix_case_insensitive(after_revoke.trim_start(), "GRANT OPTION FOR")
            .is_some()
            || find_keyword_outside_quotes(after_revoke, "GRANT OPTION").is_some()
        {
            return Err(ParseError::InvalidRelationalSql);
        }
        let on_idx = find_keyword_outside_quotes(after_revoke, "ON")
            .ok_or(ParseError::InvalidRelationalSql)?;
        let (privileges, target_and_grantee) = after_revoke.split_at(on_idx);
        let target_and_grantee = strip_keyword_prefix_case_insensitive(target_and_grantee, "ON")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim_start();
        let from_idx = find_keyword_outside_quotes(target_and_grantee, "FROM")
            .ok_or(ParseError::InvalidRelationalSql)?;
        let (target, grantee) = target_and_grantee.split_at(from_idx);
        if !target.trim().eq_ignore_ascii_case("TABLES") {
            return Err(ParseError::InvalidRelationalSql);
        }
        let grantee = strip_keyword_prefix_case_insensitive(grantee, "FROM")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim();
        return Ok(Command::RevokeDefaultTablePrivileges(
            DefaultTablePrivileges {
                grantee: parse_acl_grantee(grantee)?,
                privileges: parse_table_privileges(privileges)?,
            },
        ));
    }

    Err(ParseError::InvalidRelationalSql)
}

fn parse_acl_relation_target(target: &str) -> Result<(String, AclRelationKind), ParseError> {
    let mut target = target.trim();
    let mut kind = AclRelationKind::Relation;
    if let Some(after_materialized) = strip_keyword_prefix_case_insensitive(target, "MATERIALIZED")
    {
        target = strip_keyword_prefix_case_insensitive(after_materialized.trim_start(), "VIEW")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim_start();
        kind = AclRelationKind::MaterializedView;
    } else if let Some(after_sequence) = strip_keyword_prefix_case_insensitive(target, "SEQUENCE") {
        target = after_sequence.trim_start();
        kind = AclRelationKind::Sequence;
    } else if let Some(after_table) = strip_keyword_prefix_case_insensitive(target, "TABLE") {
        target = after_table.trim_start();
        kind = AclRelationKind::Table;
    } else if let Some(after_view) = strip_keyword_prefix_case_insensitive(target, "VIEW") {
        target = after_view.trim_start();
        kind = AclRelationKind::View;
    }
    if let Some(after_table) = strip_keyword_prefix_case_insensitive(target, "TABLE") {
        target = after_table.trim_start();
    }
    if target.is_empty()
        || target.contains(',')
        || find_keyword_outside_quotes(target, "COLUMN").is_some()
        || find_keyword_outside_quotes(target, "SCHEMA").is_some()
        || find_keyword_outside_quotes(target, "SEQUENCE").is_some()
        || find_keyword_outside_quotes(target, "FUNCTION").is_some()
        || find_keyword_outside_quotes(target, "VIEW").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok((normalize_relation_identifier(target)?, kind))
}

fn parse_acl_grantee(grantee: &str) -> Result<String, ParseError> {
    let trimmed = grantee.trim();
    if trimmed.contains(',') || trimmed.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let normalized = normalize_identifier(trimmed)?;
    Ok(normalized)
}

fn parse_table_privileges(input: &str) -> Result<Vec<TablePrivilege>, ParseError> {
    let trimmed = input.trim();
    if trimmed.eq_ignore_ascii_case("ALL") || trimmed.eq_ignore_ascii_case("ALL PRIVILEGES") {
        return Ok(vec![
            TablePrivilege::Select,
            TablePrivilege::Insert,
            TablePrivilege::Update,
            TablePrivilege::Delete,
        ]);
    }
    if trimmed.contains('(') || trimmed.contains(')') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let mut privileges = Vec::new();
    for token in split_csv(trimmed)? {
        let privilege = match token.trim().to_ascii_uppercase().as_str() {
            "SELECT" => TablePrivilege::Select,
            "INSERT" => TablePrivilege::Insert,
            "UPDATE" => TablePrivilege::Update,
            "DELETE" => TablePrivilege::Delete,
            _ => return Err(ParseError::InvalidRelationalSql),
        };
        if !privileges.contains(&privilege) {
            privileges.push(privilege);
        }
    }
    if privileges.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(privileges)
}

fn parse_schema_privileges(input: &str) -> Result<Vec<SchemaPrivilege>, ParseError> {
    let trimmed = input.trim();
    if trimmed.eq_ignore_ascii_case("ALL") || trimmed.eq_ignore_ascii_case("ALL PRIVILEGES") {
        return Ok(vec![SchemaPrivilege::Usage, SchemaPrivilege::Create]);
    }
    if trimmed.contains('(') || trimmed.contains(')') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let mut privileges = Vec::new();
    for token in split_csv(trimmed)? {
        let privilege = match token.trim().to_ascii_uppercase().as_str() {
            "USAGE" => SchemaPrivilege::Usage,
            "CREATE" => SchemaPrivilege::Create,
            _ => return Err(ParseError::InvalidRelationalSql),
        };
        if !privileges.contains(&privilege) {
            privileges.push(privilege);
        }
    }
    if privileges.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(privileges)
}

fn parse_database_privileges(input: &str) -> Result<Vec<DatabasePrivilege>, ParseError> {
    let trimmed = input.trim();
    if trimmed.eq_ignore_ascii_case("ALL") || trimmed.eq_ignore_ascii_case("ALL PRIVILEGES") {
        return Ok(vec![
            DatabasePrivilege::Connect,
            DatabasePrivilege::Temporary,
        ]);
    }
    if trimmed.contains('(') || trimmed.contains(')') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let mut privileges = Vec::new();
    for token in split_csv(trimmed)? {
        let privilege = match token.trim().to_ascii_uppercase().as_str() {
            "CONNECT" => DatabasePrivilege::Connect,
            "TEMP" | "TEMPORARY" => DatabasePrivilege::Temporary,
            _ => return Err(ParseError::InvalidRelationalSql),
        };
        if !privileges.contains(&privilege) {
            privileges.push(privilege);
        }
    }
    if privileges.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(privileges)
}

fn parse_tablespace_privileges(input: &str) -> Result<Vec<TablespacePrivilege>, ParseError> {
    let trimmed = input.trim();
    if trimmed.eq_ignore_ascii_case("ALL") || trimmed.eq_ignore_ascii_case("ALL PRIVILEGES") {
        return Ok(vec![TablespacePrivilege::Create]);
    }
    if trimmed.contains('(') || trimmed.contains(')') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let mut privileges = Vec::new();
    for token in split_csv(trimmed)? {
        let privilege = match token.trim().to_ascii_uppercase().as_str() {
            "CREATE" => TablespacePrivilege::Create,
            _ => return Err(ParseError::InvalidRelationalSql),
        };
        if !privileges.contains(&privilege) {
            privileges.push(privilege);
        }
    }
    if privileges.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(privileges)
}

fn parse_function_privileges(input: &str) -> Result<Vec<FunctionPrivilege>, ParseError> {
    let trimmed = input.trim();
    if trimmed.eq_ignore_ascii_case("ALL") || trimmed.eq_ignore_ascii_case("ALL PRIVILEGES") {
        return Ok(vec![FunctionPrivilege::Execute]);
    }
    if trimmed.contains('(') || trimmed.contains(')') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let mut privileges = Vec::new();
    for token in split_csv(trimmed)? {
        let privilege = match token.trim().to_ascii_uppercase().as_str() {
            "EXECUTE" => FunctionPrivilege::Execute,
            _ => return Err(ParseError::InvalidRelationalSql),
        };
        if !privileges.contains(&privilege) {
            privileges.push(privilege);
        }
    }
    if privileges.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(privileges)
}
