//! Neutral SQL vocabulary for the GPU database.
//!
//! This crate holds the command AST (`Command` and the DDL/DML structs),
//! `SqlType`/`SqlValue`, the `COPY` helpers, and the SQL parser (`parse_command`
//! and friends) together with [`ParseError`]. It is wire-agnostic: the pgwire
//! framing and codecs live in `gpu_db_protocol`, which re-exports everything
//! here so its public API is unchanged. Keeping the vocabulary in a lower crate
//! lets `gpu_db_engine` consume parsed SQL without depending on the wire crate
//! (roadmap §9.2: invert the engine→protocol dependency).

mod acl;
mod ast;
mod command;
mod copy;

pub use acl::{
    AclRelationKind, DatabasePrivilege, DatabasePrivileges, DefaultTablePrivileges,
    FunctionPrivilege, FunctionPrivileges, GrantTable, RevokeTable, SchemaPrivilege,
    SchemaPrivileges, TablePrivilege, TablespacePrivilege, TablespacePrivileges,
};
pub use ast::{
    AddCheckConstraint, AddColumn, AddForeignKey, AddPrimaryKey, AddUniqueConstraint,
    AlterColumnDefault, CheckConstraint, ColumnDef, ColumnDefault, Command, CommentOn,
    CommentTarget, CreateDatabase, CreateDomain, CreateExtension, CreateFunction, CreateIndex,
    CreateMaterializedView, CreatePublication, CreateRole, CreateSchema, CreateSequence,
    CreateSubscription, CreateTable, CreateTablespace, CreateView, Delete, DropColumn,
    DropConstraint, DropDatabase, DropDomain, DropExtension, DropFunction, DropIndex,
    DropMaterializedView, DropPublication, DropRole, DropSchema, DropSequence, DropSubscription,
    DropTable, DropTablespace, DropView, Insert, PrimaryKey, PublicationTarget,
    RefreshMaterializedView, RenameColumn, RenameConstraint, RenameDatabase, RenameFunction,
    RenameIndex, RenameMaterializedView, RenameRole, RenameSequence, RenameTable, RenameTablespace,
    RenameView, SelectFunction, SequenceCurrVal, SequenceNextVal, SequenceSetVal, TruncateTable,
    UniqueConstraint, Update, UpdateAssignment,
};
pub use command::{parse_command, parse_command_allowing_catalog};
pub use copy::{
    is_copy_statement, is_supported_extended_copy, parse_copy_from_stdin, parse_copy_row,
    parse_copy_to_stdout_table, CopyColumn, CopyFormat, CopyFromStdin, CopyOptions, CopyParseError,
    CopyToStdout,
};
mod decimal;
mod relation;
mod scalar;
mod select;

pub use decimal::{Decimal128, NumericOverflow};
pub mod datetime;
pub use scalar::{
    SqlType, SqlValue, NUMERIC_DEFAULT_PRECISION, NUMERIC_DEFAULT_SCALE, SUPPORTED_SQL_TYPES,
};
pub use select::{
    GroupedAggKind, GroupedAggregate, Select, SelectFilter, SelectFilterOp, SelectOrder,
    SelectProjection,
};
pub mod uuid;

use relation::parse_relational_command;
use scalar::{
    parse_bool_value, parse_sql_value, parse_supported_sql_type_name, parse_typed_value_from_str,
};
use select::{parse_select, parse_select_filter, parse_select_filter_groups};

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("empty command")]
    Empty,
    #[error("unsupported command: {0}")]
    Unsupported(String),
    #[error("invalid SET syntax; expected: SET key=value or SET key TO value")]
    InvalidSet,
    #[error("invalid DEL/DELETE syntax; expected: DEL key or DELETE [FROM] key")]
    InvalidDel,
    #[error("invalid GET syntax; expected: GET key")]
    InvalidGet,
    #[error("invalid relational SQL syntax; supported subset: CREATE TABLE name (...), CREATE [UNIQUE] INDEX name ON table (column), DROP INDEX [IF EXISTS] name, INSERT INTO name (...) VALUES (...), UPDATE name SET column = literal [, ...] WHERE column (=|<|<=|>|>=) literal | column BETWEEN literal AND literal | column IN (literal, ...) | text_column LIKE 'prefix%' [AND ...] [OR ...], DELETE FROM name WHERE column (=|<|<=|>|>=) literal | column BETWEEN literal AND literal | column IN (literal, ...) | text_column LIKE 'prefix%' [AND ...] [OR ...], SELECT [DISTINCT] columns|COUNT(*)|SUM(int4_column)|AVG(int4_column)|MIN(column)|MAX(column)|column, COUNT(*)|column, SUM(int4_column)|column, AVG(int4_column)|column, MIN(column)|column, MAX(column) FROM name [WHERE column (=|<|<=|>|>=) literal | column BETWEEN literal AND literal | column IN (literal, ...) | text_column LIKE 'prefix%' [AND ...] [OR ...]] [GROUP BY column] [HAVING grouped_column|count|sum|avg|min|max (=|<|<=|>|>=) literal [AND ...] [OR ...]] [ORDER BY selected_column|count|sum|avg|min|max [ASC|DESC]] [LIMIT n] [OFFSET n]")]
    InvalidRelationalSql,
    #[error("LIMIT must not be negative")]
    NegativeLimit,
    #[error("OFFSET must not be negative")]
    NegativeOffset,
    #[error("invalid RESET/DISCARD/DEALLOCATE/CLOSE/LISTEN/NOTIFY/UNLISTEN syntax; expected: RESET ALL|ROLE|AUTHORIZATION|AUTH|SESSION AUTHORIZATION[ [TO] DEFAULT]|SESSION AUTH[ [TO] DEFAULT], DISCARD {{ALL|TEMP|TEMPORARY|TEMP TABLES|TEMPORARY TABLES|PLANS|SEQUENCES}}, DEALLOCATE {{ALL|name|PREPARE|PREPARED name}}, CLOSE {{ALL|name}}, LISTEN channel, NOTIFY channel[, payload], or UNLISTEN [*|ALL|channel]")]
    InvalidReset,
}

fn strip_keyword_prefix_case_insensitive<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    if input.len() < keyword.len() || !input[..keyword.len()].eq_ignore_ascii_case(keyword) {
        return None;
    }
    if input.len() > keyword.len()
        && !input[keyword.len()..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        return None;
    }
    Some(&input[keyword.len()..])
}

fn strip_keyword_suffix_case_insensitive<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let trimmed = input.trim_end();
    if trimmed.len() < keyword.len()
        || !trimmed[trimmed.len() - keyword.len()..].eq_ignore_ascii_case(keyword)
    {
        return None;
    }
    let before = &trimmed[..trimmed.len() - keyword.len()];
    if !before.chars().last().is_some_and(char::is_whitespace) {
        return None;
    }
    Some(before)
}

fn parse_select_function(input: &str) -> Result<Command, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "SELECT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if find_keyword_outside_quotes(rest, "FROM").is_some()
        || find_keyword_outside_quotes(rest, "WHERE").is_some()
        || find_keyword_outside_quotes(rest, "ORDER").is_some()
        || find_keyword_outside_quotes(rest, "GROUP").is_some()
        || find_keyword_outside_quotes(rest, "LIMIT").is_some()
        || find_keyword_outside_quotes(rest, "OFFSET").is_some()
        || rest.contains(',')
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = rest.rfind(')').ok_or(ParseError::InvalidRelationalSql)?;
    if close + 1 != rest.len() || !rest[open + 1..close].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let name = normalize_function_signature(rest)?;
    if name == "current_schema" {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(Command::SelectFunction(SelectFunction { name }))
}

fn parse_sequence_value_function(input: &str) -> Result<Command, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "SELECT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if find_keyword_outside_quotes(rest, "FROM").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(rest, open).ok_or(ParseError::InvalidRelationalSql)?;
    if !rest[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let function = rest[..open].trim();
    let function = function
        .strip_prefix("pg_catalog.")
        .or_else(|| function.strip_prefix("PG_CATALOG."))
        .unwrap_or(function);
    let args = split_csv(&rest[open + 1..close])?;
    match function.to_ascii_lowercase().as_str() {
        "nextval" => {
            let [target] = args.as_slice() else {
                return Err(ParseError::InvalidRelationalSql);
            };
            Ok(Command::SequenceNextVal(SequenceNextVal {
                name: parse_sequence_regclass_arg(target.trim())?,
            }))
        }
        "currval" => {
            let [target] = args.as_slice() else {
                return Err(ParseError::InvalidRelationalSql);
            };
            Ok(Command::SequenceCurrVal(SequenceCurrVal {
                name: parse_sequence_regclass_arg(target.trim())?,
            }))
        }
        "setval" => {
            let ([target, value] | [target, value, _]) = args.as_slice() else {
                return Err(ParseError::InvalidRelationalSql);
            };
            let is_called = if args.len() == 3 {
                parse_bool_literal(args[2].trim())?
            } else {
                true
            };
            Ok(Command::SequenceSetVal(SequenceSetVal {
                name: parse_sequence_regclass_arg(target.trim())?,
                value: parse_i64_literal(value.trim())?,
                is_called,
            }))
        }
        _ => Err(ParseError::InvalidRelationalSql),
    }
}

fn parse_sequence_regclass_arg(input: &str) -> Result<String, ParseError> {
    let literal = input
        .split_once("::")
        .map(|(literal, cast)| {
            let cast = cast.trim();
            if cast.eq_ignore_ascii_case("regclass")
                || cast.eq_ignore_ascii_case("pg_catalog.regclass")
            {
                Ok(literal.trim())
            } else {
                Err(ParseError::InvalidRelationalSql)
            }
        })
        .unwrap_or(Ok(input.trim()))?;
    let SqlValue::Text(name) = parse_sql_value(literal)? else {
        return Err(ParseError::InvalidRelationalSql);
    };
    normalize_relation_identifier(&name)
}

fn parse_i64_literal(input: &str) -> Result<i64, ParseError> {
    input
        .parse::<i64>()
        .map_err(|_| ParseError::InvalidRelationalSql)
}

fn parse_bool_literal(input: &str) -> Result<bool, ParseError> {
    if input.eq_ignore_ascii_case("true") || input.eq_ignore_ascii_case("t") {
        Ok(true)
    } else if input.eq_ignore_ascii_case("false") || input.eq_ignore_ascii_case("f") {
        Ok(false)
    } else {
        Err(ParseError::InvalidRelationalSql)
    }
}

fn parse_comment_on(input: &str) -> Result<CommentOn, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "COMMENT")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "ON"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (target, rest) = if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "DATABASE")
    {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let database = normalize_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Database { database },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "ROLE") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let role = normalize_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Role { role },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "SCHEMA") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let schema = normalize_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Schema { schema },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "TABLESPACE") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let tablespace = normalize_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Tablespace { tablespace },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "TABLE") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let table = normalize_relation_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Table { table },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "COLUMN") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let target = rest[..is_pos].trim();
        let (table, column) = target
            .rsplit_once('.')
            .ok_or(ParseError::InvalidRelationalSql)?;
        let table = normalize_relation_identifier(table.trim())?;
        let column = normalize_identifier(column.trim())?;
        (
            CommentTarget::Column { table, column },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "INDEX") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let index = normalize_relation_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Index { index },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "VIEW") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let view = normalize_relation_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::View { view },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "MATERIALIZED") {
        let rest = strip_keyword_prefix_case_insensitive(rest.trim_start(), "VIEW")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let materialized_view = normalize_relation_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::MaterializedView { materialized_view },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "FUNCTION") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let function = normalize_function_signature(rest[..is_pos].trim())?;
        (
            CommentTarget::Function { function },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "EXTENSION") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let extension = normalize_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Extension { extension },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "SEQUENCE") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let sequence = normalize_relation_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Sequence { sequence },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "DOMAIN") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let domain = normalize_relation_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Domain { domain },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "PUBLICATION") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let publication = normalize_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Publication { publication },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "SUBSCRIPTION") {
        let rest = rest.trim_start();
        let is_pos =
            find_keyword_outside_quotes(rest, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let subscription = normalize_identifier(rest[..is_pos].trim())?;
        (
            CommentTarget::Subscription { subscription },
            rest[is_pos + "IS".len()..].trim(),
        )
    } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "CONSTRAINT") {
        let rest = rest.trim_start();
        let on_pos =
            find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
        let constraint = normalize_identifier(rest[..on_pos].trim())?;
        let after_on = rest[on_pos + "ON".len()..].trim_start();
        let is_pos =
            find_keyword_outside_quotes(after_on, "IS").ok_or(ParseError::InvalidRelationalSql)?;
        let table = normalize_relation_identifier(after_on[..is_pos].trim())?;
        (
            CommentTarget::Constraint { table, constraint },
            after_on[is_pos + "IS".len()..].trim(),
        )
    } else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let comment = if rest.eq_ignore_ascii_case("NULL") {
        None
    } else {
        match parse_sql_value(rest)? {
            SqlValue::Text(value) => Some(value),
            _ => return Err(ParseError::InvalidRelationalSql),
        }
    };
    Ok(CommentOn { target, comment })
}

fn parse_create_view(input: &str) -> Result<CreateView, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (rest, or_replace) = if let Some(after_or) =
        strip_keyword_prefix_case_insensitive(rest, "OR")
    {
        let after_replace = strip_keyword_prefix_case_insensitive(after_or.trim_start(), "REPLACE")
            .ok_or(ParseError::InvalidRelationalSql)?;
        (after_replace.trim_start(), true)
    } else {
        (rest, false)
    };
    let rest = strip_keyword_prefix_case_insensitive(rest, "VIEW")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "IF").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "TEMP").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "TEMPORARY").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "MATERIALIZED").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let as_pos = find_keyword_outside_quotes(rest, "AS").ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_relation_identifier(rest[..as_pos].trim())?;
    let definition = rest[as_pos + "AS".len()..].trim();
    if definition.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if find_keyword_outside_quotes(definition, "WITH").is_some()
        && definition.to_ascii_uppercase().contains("CHECK OPTION")
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let query = parse_select(definition, false)?;
    if query.table == name {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreateView {
        name,
        query,
        definition: definition.to_string(),
        or_replace,
    })
}

fn parse_create_sequence(input: &str) -> Result<CreateSequence, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SEQUENCE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if rest.is_empty()
        || strip_keyword_prefix_case_insensitive(rest, "IF").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "TEMP").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "TEMPORARY").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let mut parts = rest.split_whitespace();
    let Some(name) = parts.next() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let suffix = parts.collect::<Vec<_>>().join(" ");
    if !suffix.is_empty()
        && !suffix
            .eq_ignore_ascii_case("START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1")
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreateSequence {
        name: normalize_relation_identifier(name)?,
    })
}

fn parse_create_function(input: &str) -> Result<CreateFunction, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "FUNCTION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let returns_pos =
        find_keyword_outside_quotes(rest, "RETURNS").ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_function_signature(rest[..returns_pos].trim())?;
    let rest = rest[returns_pos + "RETURNS".len()..].trim_start();
    let language_pos =
        find_keyword_outside_quotes(rest, "LANGUAGE").ok_or(ParseError::InvalidRelationalSql)?;
    let return_type = parse_supported_sql_type_name(rest[..language_pos].trim())
        .ok_or(ParseError::InvalidRelationalSql)?;
    let rest = rest[language_pos + "LANGUAGE".len()..].trim_start();
    let as_pos = find_keyword_outside_quotes(rest, "AS").ok_or(ParseError::InvalidRelationalSql)?;
    let language = normalize_identifier(rest[..as_pos].trim())?;
    if language != "sql" {
        return Err(ParseError::InvalidRelationalSql);
    }
    let raw_body = rest[as_pos + "AS".len()..].trim();
    let body = if let Some(value) = parse_dollar_quoted_literal(raw_body) {
        value
    } else {
        match parse_sql_value(raw_body)? {
            SqlValue::Text(value) => value,
            _ => return Err(ParseError::InvalidRelationalSql),
        }
    };
    Ok(CreateFunction {
        name,
        return_type,
        body,
    })
}

fn parse_dollar_quoted_literal(input: &str) -> Option<String> {
    let input = input.trim();
    let after_open = input.strip_prefix('$')?;
    let tag_end = after_open.find('$')?;
    let tag = &after_open[..tag_end];
    if !tag
        .chars()
        .all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return None;
    }
    let delimiter = format!("${tag}$");
    let body_start = delimiter.len();
    let body_end = input[body_start..].find(&delimiter)? + body_start;
    if !input[body_end + delimiter.len()..].trim().is_empty() {
        return None;
    }
    Some(input[body_start..body_end].to_string())
}

fn parse_create_materialized_view(input: &str) -> Result<CreateMaterializedView, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "MATERIALIZED"))
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "VIEW"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "IF").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "TEMP").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "TEMPORARY").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let as_pos = find_keyword_outside_quotes(rest, "AS").ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_relation_identifier(rest[..as_pos].trim())?;
    let mut definition = rest[as_pos + "AS".len()..].trim();
    let mut with_data = true;
    if let Some(with_pos) = find_keyword_outside_quotes(definition, "WITH") {
        let options = definition[with_pos + "WITH".len()..].trim();
        if options.eq_ignore_ascii_case("DATA") {
            with_data = true;
        } else if options.eq_ignore_ascii_case("NO DATA") {
            with_data = false;
        } else {
            return Err(ParseError::InvalidRelationalSql);
        }
        definition = definition[..with_pos].trim_end();
    }
    if definition.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let query = parse_select(definition, false)?;
    if query.table == name {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreateMaterializedView {
        name,
        query,
        definition: definition.to_string(),
        with_data,
    })
}

fn parse_refresh_materialized_view(input: &str) -> Result<RefreshMaterializedView, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "REFRESH")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "MATERIALIZED"))
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "VIEW"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if let Some(after_concurrently) = strip_keyword_prefix_case_insensitive(rest, "CONCURRENTLY") {
        let _ = after_concurrently;
        return Err(ParseError::InvalidRelationalSql);
    }
    if rest.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let with_pos = find_keyword_outside_quotes(rest, "WITH");
    if let Some(with_pos) = with_pos {
        let options = rest[with_pos + "WITH".len()..].trim();
        if !options.eq_ignore_ascii_case("DATA") {
            return Err(ParseError::InvalidRelationalSql);
        }
        rest = rest[..with_pos].trim_end();
    }
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RefreshMaterializedView {
        name: normalize_relation_identifier(rest)?,
    })
}

fn parse_rename_view(input: &str) -> Result<RenameView, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "VIEW"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "IF").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "ALL").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "CURRENT").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "MATERIALIZED").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_relation_identifier(rest[..rename_pos].trim())?;
    let after_rename = rest[rename_pos + "RENAME".len()..].trim_start();
    let after_to = strip_keyword_prefix_case_insensitive(after_rename, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if after_to.is_empty()
        || find_keyword_outside_quotes(after_to, "CASCADE").is_some()
        || find_keyword_outside_quotes(after_to, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameView {
        old_name,
        new_name: normalize_identifier(after_to)?,
    })
}

fn parse_rename_sequence(input: &str) -> Result<RenameSequence, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SEQUENCE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "IF").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "ALL").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "CURRENT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_relation_identifier(rest[..rename_pos].trim())?;
    let after_rename = rest[rename_pos + "RENAME".len()..].trim_start();
    let after_to = strip_keyword_prefix_case_insensitive(after_rename, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if after_to.is_empty()
        || find_keyword_outside_quotes(after_to, "CASCADE").is_some()
        || find_keyword_outside_quotes(after_to, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameSequence {
        old_name,
        new_name: normalize_identifier(after_to)?,
    })
}

fn parse_rename_function(input: &str) -> Result<RenameFunction, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "FUNCTION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "IF").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "ALL").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "CURRENT").is_some()
        || find_keyword_outside_quotes(rest, "OWNER").is_some()
        || find_keyword_outside_quotes(rest, "SET").is_some()
        || find_keyword_outside_quotes(rest, "RESET").is_some()
        || find_keyword_outside_quotes(rest, "DEPENDS").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_function_signature(rest[..rename_pos].trim())?;
    let after_rename = rest[rename_pos + "RENAME".len()..].trim_start();
    let after_to = strip_keyword_prefix_case_insensitive(after_rename, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if after_to.is_empty()
        || find_keyword_outside_quotes(after_to, "CASCADE").is_some()
        || find_keyword_outside_quotes(after_to, "RESTRICT").is_some()
        || after_to.contains('.')
        || after_to.contains('(')
        || after_to.contains(')')
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameFunction {
        old_name,
        new_name: normalize_identifier(after_to)?,
    })
}

fn parse_rename_materialized_view(input: &str) -> Result<RenameMaterializedView, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "MATERIALIZED"))
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "VIEW"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "IF").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "ALL").is_some()
        || strip_keyword_prefix_case_insensitive(rest, "CURRENT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_relation_identifier(rest[..rename_pos].trim())?;
    let after_rename = rest[rename_pos + "RENAME".len()..].trim_start();
    let after_to = strip_keyword_prefix_case_insensitive(after_rename, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if after_to.is_empty()
        || find_keyword_outside_quotes(after_to, "CASCADE").is_some()
        || find_keyword_outside_quotes(after_to, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameMaterializedView {
        old_name,
        new_name: normalize_identifier(after_to)?,
    })
}

fn parse_drop_view(input: &str) -> Result<DropView, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "VIEW"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "MATERIALIZED").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty() || find_keyword_outside_quotes(rest, "CASCADE").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if find_keyword_outside_quotes(rest, "RESTRICT").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let views = split_csv(rest)?;
    if views.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropView {
        names: views
            .iter()
            .map(|view| normalize_relation_identifier(view.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_drop_sequence(input: &str) -> Result<DropSequence, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SEQUENCE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let sequences = split_csv(rest)?;
    if sequences.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropSequence {
        names: sequences
            .into_iter()
            .map(|sequence| normalize_relation_identifier(sequence.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_create_publication(input: &str) -> Result<CreatePublication, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "PUBLICATION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (raw_name, rest) = split_leading_identifier(rest)?;
    let name = normalize_identifier(raw_name)?;
    let rest = rest.trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "FOR")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if let Some(after_all) = strip_keyword_prefix_case_insensitive(rest, "ALL") {
        let after_tables = strip_keyword_prefix_case_insensitive(after_all.trim_start(), "TABLES")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim();
        if !after_tables.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        return Ok(CreatePublication {
            name,
            target: PublicationTarget::AllTables,
        });
    }
    let rest = strip_keyword_prefix_case_insensitive(rest, "TABLE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "WHERE").is_some()
        || find_keyword_outside_quotes(rest, "WITH").is_some()
        || find_keyword_outside_quotes(rest, "ONLY").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let tables = split_csv(rest)?;
    if tables.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreatePublication {
        name,
        target: PublicationTarget::Tables(
            tables
                .into_iter()
                .map(|table| normalize_relation_identifier(table.trim()))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

fn parse_drop_publication(input: &str) -> Result<DropPublication, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "PUBLICATION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let publications = split_csv(rest)?;
    if publications.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropPublication {
        names: publications
            .into_iter()
            .map(|publication| normalize_identifier(publication.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_create_subscription(input: &str) -> Result<CreateSubscription, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SUBSCRIPTION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (raw_name, rest) = split_leading_identifier(rest)?;
    let name = normalize_identifier(raw_name)?;
    let rest = strip_keyword_prefix_case_insensitive(rest.trim_start(), "CONNECTION")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let publication_idx =
        find_keyword_outside_quotes(rest, "PUBLICATION").ok_or(ParseError::InvalidRelationalSql)?;
    let (connection_literal, after_connection) = rest.split_at(publication_idx);
    let Ok(SqlValue::Text(connection)) = parse_sql_value(connection_literal.trim()) else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let rest = strip_keyword_prefix_case_insensitive(after_connection, "PUBLICATION")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (publication_list, option_clause) =
        if let Some(with_idx) = find_keyword_outside_quotes(rest, "WITH") {
            let (publications, options) = rest.split_at(with_idx);
            (
                publications.trim(),
                Some(
                    strip_keyword_prefix_case_insensitive(options, "WITH")
                        .ok_or(ParseError::InvalidRelationalSql)?
                        .trim(),
                ),
            )
        } else {
            (rest.trim(), None)
        };
    if publication_list.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let publications = split_csv(publication_list)?
        .into_iter()
        .map(|publication| normalize_identifier(publication.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    if publications.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let Some(options) = option_clause else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let options = options
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or(ParseError::InvalidRelationalSql)?;
    let mut saw_connect = false;
    let mut saw_enabled = false;
    for option in split_csv(options)? {
        let (key, value) = option
            .split_once('=')
            .ok_or(ParseError::InvalidRelationalSql)?;
        let key = key.trim();
        let value = value.trim();
        if key.eq_ignore_ascii_case("connect") {
            if saw_connect || parse_bool_literal(value)? {
                return Err(ParseError::InvalidRelationalSql);
            }
            saw_connect = true;
        } else if key.eq_ignore_ascii_case("enabled") {
            if saw_enabled || parse_bool_literal(value)? {
                return Err(ParseError::InvalidRelationalSql);
            }
            saw_enabled = true;
        } else {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
    if !saw_connect || !saw_enabled {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreateSubscription {
        name,
        connection,
        publications,
    })
}

fn parse_drop_subscription(input: &str) -> Result<DropSubscription, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SUBSCRIPTION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let subscriptions = split_csv(rest)?;
    if subscriptions.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropSubscription {
        names: subscriptions
            .into_iter()
            .map(|subscription| normalize_identifier(subscription.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_create_role(input: &str) -> Result<CreateRole, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (is_user, rest) = if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "USER") {
        (true, rest.trim_start())
    } else {
        (
            false,
            strip_keyword_prefix_case_insensitive(rest, "ROLE")
                .ok_or(ParseError::InvalidRelationalSql)?
                .trim_start(),
        )
    };
    let (raw_name, rest) = split_leading_identifier(rest)?;
    let name = normalize_identifier(raw_name)?;
    let mut rest = rest.trim_start();
    if let Some(after_with) = strip_keyword_prefix_case_insensitive(rest, "WITH") {
        rest = after_with.trim_start();
    }
    if rest.is_empty() {
        return Ok(CreateRole {
            name,
            login: is_user,
        });
    }
    let mut login = is_user;
    let mut saw_login_option = false;
    for token in rest.split_whitespace() {
        if token.eq_ignore_ascii_case("LOGIN") {
            if saw_login_option {
                return Err(ParseError::InvalidRelationalSql);
            }
            login = true;
            saw_login_option = true;
        } else if token.eq_ignore_ascii_case("NOLOGIN") {
            if saw_login_option {
                return Err(ParseError::InvalidRelationalSql);
            }
            login = false;
            saw_login_option = true;
        } else {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
    Ok(CreateRole { name, login })
}

fn parse_drop_role(input: &str) -> Result<DropRole, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if let Some(after_user) = strip_keyword_prefix_case_insensitive(rest, "USER") {
        rest = after_user.trim_start();
    } else {
        rest = strip_keyword_prefix_case_insensitive(rest, "ROLE")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim_start();
    }
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let roles = split_csv(rest)?;
    if roles.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropRole {
        names: roles
            .into_iter()
            .map(|role| normalize_identifier(role.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_rename_role(input: &str) -> Result<RenameRole, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "ROLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_identifier(rest[..rename_pos].trim())?;
    let after_rename = rest[rename_pos + "RENAME".len()..].trim_start();
    let after_to = strip_keyword_prefix_case_insensitive(after_rename, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if after_to.is_empty()
        || find_keyword_outside_quotes(after_to, "WITH").is_some()
        || find_keyword_outside_quotes(after_to, "CASCADE").is_some()
        || find_keyword_outside_quotes(after_to, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameRole {
        old_name,
        new_name: normalize_identifier(after_to)?,
    })
}

fn parse_create_database(input: &str) -> Result<CreateDatabase, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "DATABASE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (raw_name, rest) = split_leading_identifier(rest)?;
    if !rest.trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreateDatabase {
        name: normalize_identifier(raw_name)?,
    })
}

fn parse_drop_database(input: &str) -> Result<DropDatabase, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "DATABASE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "FORCE").is_some()
        || find_keyword_outside_quotes(rest, "WITH").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let names = split_csv(rest)?;
    if names.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropDatabase {
        names: names
            .into_iter()
            .map(|name| normalize_identifier(name.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_rename_database(input: &str) -> Result<RenameDatabase, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "DATABASE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_identifier(rest[..rename_pos].trim())?;
    let after_rename = rest[rename_pos + "RENAME".len()..].trim_start();
    let after_to = strip_keyword_prefix_case_insensitive(after_rename, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if after_to.is_empty()
        || find_keyword_outside_quotes(after_to, "WITH").is_some()
        || find_keyword_outside_quotes(after_to, "OWNER").is_some()
        || find_keyword_outside_quotes(after_to, "SET").is_some()
        || find_keyword_outside_quotes(after_to, "CASCADE").is_some()
        || find_keyword_outside_quotes(after_to, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameDatabase {
        old_name,
        new_name: normalize_identifier(after_to)?,
    })
}

fn parse_create_tablespace(input: &str) -> Result<CreateTablespace, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "TABLESPACE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (raw_name, rest) = split_leading_identifier(rest)?;
    let name = normalize_identifier(raw_name)?;
    let rest = if let Some(after_owner) =
        strip_keyword_prefix_case_insensitive(rest.trim_start(), "OWNER")
    {
        let after_owner = after_owner.trim_start();
        let Some(after_postgres) = strip_keyword_prefix_case_insensitive(after_owner, "postgres")
        else {
            return Err(ParseError::InvalidRelationalSql);
        };
        after_postgres
    } else {
        rest
    };
    let rest = strip_keyword_prefix_case_insensitive(rest.trim_start(), "LOCATION")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let SqlValue::Text(location) = parse_sql_value(rest)? else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(CreateTablespace { name, location })
}

fn parse_drop_tablespace(input: &str) -> Result<DropTablespace, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "TABLESPACE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let names = split_csv(rest)?;
    if names.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropTablespace {
        names: names
            .into_iter()
            .map(|name| normalize_identifier(name.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_rename_tablespace(input: &str) -> Result<RenameTablespace, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLESPACE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_identifier(rest[..rename_pos].trim())?;
    let after_rename = rest[rename_pos + "RENAME".len()..].trim_start();
    let after_to = strip_keyword_prefix_case_insensitive(after_rename, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if after_to.is_empty()
        || find_keyword_outside_quotes(after_to, "WITH").is_some()
        || find_keyword_outside_quotes(after_to, "OWNER").is_some()
        || find_keyword_outside_quotes(after_to, "SET").is_some()
        || find_keyword_outside_quotes(after_to, "CASCADE").is_some()
        || find_keyword_outside_quotes(after_to, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameTablespace {
        old_name,
        new_name: normalize_identifier(after_to)?,
    })
}

fn split_leading_identifier(input: &str) -> Result<(&str, &str), ParseError> {
    let trimmed = input.trim_start();
    if trimmed.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if let Some(rest) = trimmed.strip_prefix('"') {
        let mut escaped = false;
        for (idx, ch) in rest.char_indices() {
            if ch == '"' {
                if escaped {
                    escaped = false;
                    continue;
                }
                let end = idx + 2;
                return Ok((&trimmed[..end], &trimmed[end..]));
            }
            escaped = ch == '"';
        }
        return Err(ParseError::InvalidRelationalSql);
    }
    let end = trimmed
        .char_indices()
        .find_map(|(idx, ch)| ch.is_whitespace().then_some(idx))
        .unwrap_or(trimmed.len());
    Ok((&trimmed[..end], &trimmed[end..]))
}

fn parse_drop_materialized_view(input: &str) -> Result<DropMaterializedView, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "MATERIALIZED"))
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "VIEW"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let views = split_csv(rest)?;
    if views.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropMaterializedView {
        names: views
            .into_iter()
            .map(|view| normalize_relation_identifier(view.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_create_domain(input: &str) -> Result<CreateDomain, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "DOMAIN"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let as_pos = find_keyword_outside_quotes(rest, "AS").ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_relation_identifier(rest[..as_pos].trim())?;
    let tail = rest[as_pos + "AS".len()..].trim();
    if tail.is_empty()
        || find_keyword_outside_quotes(tail, "DEFAULT").is_some()
        || find_keyword_outside_quotes(tail, "CHECK").is_some()
        || find_keyword_outside_quotes(tail, "COLLATE").is_some()
        || find_keyword_outside_quotes(tail, "NOT").is_some()
        || tail.contains('(')
        || tail.contains(')')
        || tail.contains('[')
        || tail.contains(']')
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let base_type = parse_supported_sql_type_name(tail).ok_or(ParseError::InvalidRelationalSql)?;
    Ok(CreateDomain { name, base_type })
}

fn parse_create_extension(input: &str) -> Result<CreateExtension, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "EXTENSION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_not_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_not = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "NOT")
            .ok_or(ParseError::InvalidRelationalSql)?;
        let after_exists = strip_keyword_prefix_case_insensitive(after_not.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let (name, tail) = if let Some(with_pos) = find_keyword_outside_quotes(rest, "WITH") {
        (
            normalize_identifier(rest[..with_pos].trim())?,
            rest[with_pos + "WITH".len()..].trim_start(),
        )
    } else {
        (normalize_identifier(rest)?, "")
    };
    let schema = if tail.is_empty() {
        None
    } else {
        let schema = strip_keyword_prefix_case_insensitive(tail, "SCHEMA")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim_start();
        if schema.is_empty() || find_keyword_outside_quotes(schema, "VERSION").is_some() {
            return Err(ParseError::InvalidRelationalSql);
        }
        Some(normalize_identifier(schema)?)
    };
    Ok(CreateExtension {
        name,
        if_not_exists,
        schema,
    })
}

fn parse_drop_extension(input: &str) -> Result<DropExtension, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "EXTENSION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || rest.contains(',')
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropExtension {
        name: normalize_identifier(rest)?,
        if_exists,
    })
}

fn parse_drop_domain(input: &str) -> Result<DropDomain, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "DOMAIN"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropDomain {
        domains: split_csv(rest)?
            .into_iter()
            .map(|domain| normalize_relation_identifier(domain.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_drop_function(input: &str) -> Result<DropFunction, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "FUNCTION"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF") {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let functions = split_csv(rest)?;
    if functions.len() != 1 {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropFunction {
        name: normalize_function_signature(functions[0].trim())?,
        if_exists,
    })
}

fn normalize_function_signature(signature: &str) -> Result<String, ParseError> {
    let signature = signature.trim();
    let open = signature
        .find('(')
        .ok_or(ParseError::InvalidRelationalSql)?;
    let close = signature
        .rfind(')')
        .ok_or(ParseError::InvalidRelationalSql)?;
    if close != signature.len() - 1 || !signature[open + 1..close].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    normalize_relation_identifier(signature[..open].trim())
}

fn normalize_identifier(input: &str) -> Result<String, ParseError> {
    let s = input.trim();
    if s.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if let Some(quoted) = s.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        if quoted.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        return Ok(quoted.replace("\"\"", "\""));
    }
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(ParseError::InvalidRelationalSql);
    }
    if chars.any(|ch| !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())) {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(s.to_ascii_lowercase())
}

fn normalize_relation_identifier(input: &str) -> Result<String, ParseError> {
    let s = input.trim();
    if let Some((schema, table)) = s.split_once('.') {
        if normalize_identifier(schema)? != "public" {
            return Err(ParseError::InvalidRelationalSql);
        }
        return normalize_identifier(table);
    }
    normalize_identifier(s)
}

/// The system catalog schemas the engine answers natively (Phase-3 M2). A SELECT may
/// reference these schema-qualified; the qualifier is PRESERVED in the normalized name so
/// the engine routes the relation to its catalog synthesizer instead of a user table.
pub const PG_CATALOG_SCHEMA: &str = "pg_catalog";
pub const INFORMATION_SCHEMA: &str = "information_schema";

/// Relation-name normalizer for a SELECT's FROM target. Identical to
/// [`normalize_relation_identifier`] for user relations (`public.t`/`t` → bare `t`), but
/// PRESERVES a `pg_catalog.`/`information_schema.` qualifier (lowercased, as
/// `pg_catalog.pg_class`) so catalog relations survive parsing and reach the engine
/// instead of being rejected. DML keeps the strict (public-only) normalizer.
fn normalize_select_relation_identifier(input: &str) -> Result<String, ParseError> {
    let s = input.trim();
    if let Some((schema, table)) = s.split_once('.') {
        let schema_norm = normalize_identifier(schema)?;
        if schema_norm == PG_CATALOG_SCHEMA || schema_norm == INFORMATION_SCHEMA {
            return Ok(format!("{schema_norm}.{}", normalize_identifier(table)?));
        }
    }
    normalize_relation_identifier(s)
}

fn split_csv(input: &str) -> Result<Vec<&str>, ParseError> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    let mut in_quote = false;
    let bytes = input.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\'' => {
                if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                    idx += 1;
                } else {
                    in_quote = !in_quote;
                }
            }
            b'(' if !in_quote => depth += 1,
            b')' if !in_quote => {
                depth = depth
                    .checked_sub(1)
                    .ok_or(ParseError::InvalidRelationalSql)?;
            }
            b',' if !in_quote && depth == 0 => {
                let part = input[start..idx].trim();
                if part.is_empty() {
                    return Err(ParseError::InvalidRelationalSql);
                }
                parts.push(part);
                start = idx + 1;
            }
            _ => {}
        }
        idx += 1;
    }
    if in_quote || depth != 0 {
        return Err(ParseError::InvalidRelationalSql);
    }
    let part = input[start..].trim();
    if part.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    parts.push(part);
    Ok(parts)
}

fn find_matching_paren(input: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_quote = false;
    let bytes = input.as_bytes();
    let mut idx = open;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\'' => {
                if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                    idx += 2;
                    continue;
                }
                in_quote = !in_quote;
            }
            b'(' if !in_quote => depth += 1,
            b')' if !in_quote => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
        idx += 1;
    }
    None
}

fn find_char_outside_quotes(input: &str, needle: char) -> Option<usize> {
    let mut in_quote = false;
    let mut depth = 0usize;
    for (idx, ch) in input.char_indices() {
        if ch == '\'' {
            in_quote = !in_quote;
            continue;
        }
        if !in_quote {
            if ch == needle && depth == 0 {
                return Some(idx);
            }
            match ch {
                '(' => depth += 1,
                ')' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
    }
    None
}

fn find_keyword_outside_quotes(input: &str, keyword: &str) -> Option<usize> {
    let lower = input.to_ascii_lowercase();
    let keyword = keyword.to_ascii_lowercase();
    let bytes = input.as_bytes();
    let mut in_quote = false;
    let mut depth = 0usize;
    let mut idx = 0;
    while idx + keyword.len() <= bytes.len() {
        if bytes[idx] == b'\'' {
            if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                idx += 2;
                continue;
            }
            in_quote = !in_quote;
            idx += 1;
            continue;
        }
        match bytes[idx] {
            b'(' if !in_quote => {
                depth += 1;
                idx += 1;
                continue;
            }
            b')' if !in_quote => {
                depth = depth.saturating_sub(1);
                idx += 1;
                continue;
            }
            _ => {}
        }
        if !in_quote
            && depth == 0
            && lower[idx..].starts_with(&keyword)
            && is_keyword_boundary(input, idx, keyword.len())
        {
            return Some(idx);
        }
        idx += 1;
    }
    None
}

fn is_keyword_boundary(input: &str, start: usize, len: usize) -> bool {
    let before = input[..start]
        .chars()
        .next_back()
        .is_none_or(|ch| !is_identifier_char(ch));
    let after = input[start + len..]
        .chars()
        .next()
        .is_none_or(|ch| !is_identifier_char(ch));
    before && after
}

fn next_clause_pos(input: &str) -> Option<usize> {
    ["WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET"]
        .into_iter()
        .filter_map(|keyword| find_keyword_outside_quotes(input, keyword))
        .min()
}

fn split_keyword_chain_outside_quotes<'a>(
    mut input: &'a str,
    keyword: &str,
) -> Result<Vec<&'a str>, ParseError> {
    let mut parts = Vec::new();
    while let Some(pos) = find_keyword_outside_quotes(input, keyword) {
        let part = input[..pos].trim();
        if part.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        parts.push(part);
        input = input[pos + keyword.len()..].trim_start();
    }
    let tail = input.trim();
    if tail.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    parts.push(tail);
    Ok(parts)
}

fn split_select_and_chain_outside_quotes(input: &str) -> Result<Vec<&str>, ParseError> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut in_quote = false;
    let mut depth = 0usize;
    let mut skip_next_and = false;
    let bytes = input.as_bytes();
    let lower = input.to_ascii_lowercase();
    let mut idx = 0usize;

    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                idx += 2;
                continue;
            }
            in_quote = !in_quote;
            idx += 1;
            continue;
        }
        match bytes[idx] {
            b'(' if !in_quote => {
                depth += 1;
                idx += 1;
                continue;
            }
            b')' if !in_quote => {
                depth = depth.saturating_sub(1);
                idx += 1;
                continue;
            }
            _ => {}
        }
        if !in_quote
            && depth == 0
            && lower[idx..].starts_with("between")
            && is_keyword_boundary(input, idx, "between".len())
        {
            skip_next_and = true;
            idx += "between".len();
            continue;
        }
        if !in_quote
            && depth == 0
            && lower[idx..].starts_with("and")
            && is_keyword_boundary(input, idx, "and".len())
        {
            if skip_next_and {
                skip_next_and = false;
                idx += "and".len();
                continue;
            }
            let part = input[start..idx].trim();
            if part.is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            parts.push(part);
            idx += "and".len();
            start = idx;
            continue;
        }
        idx += 1;
    }

    let tail = input[start..].trim();
    if tail.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    parts.push(tail);
    Ok(parts)
}

fn is_identifier_char(ch: char) -> bool {
    ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()
}
