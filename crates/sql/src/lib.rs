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
mod copy;

pub use acl::{
    AclRelationKind, DatabasePrivilege, DatabasePrivileges, DefaultTablePrivileges,
    FunctionPrivilege, FunctionPrivileges, GrantTable, RevokeTable, SchemaPrivilege,
    SchemaPrivileges, TablePrivilege, TablespacePrivilege, TablespacePrivileges,
};
pub use copy::{
    is_copy_statement, is_supported_extended_copy, parse_copy_from_stdin, parse_copy_row,
    parse_copy_to_stdout_table, CopyColumn, CopyFormat, CopyFromStdin, CopyOptions, CopyParseError,
    CopyToStdout,
};
mod decimal;
mod select;

pub use decimal::{Decimal128, NumericOverflow};
pub mod datetime;
pub use select::{
    GroupedAggKind, GroupedAggregate, Select, SelectFilter, SelectFilterOp, SelectOrder,
    SelectProjection,
};
pub mod uuid;

use select::{parse_select, parse_select_filter, parse_select_filter_groups};

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Begin,
    Commit { chain: bool },
    Rollback { chain: bool },
    Flush,
    ResetAll,
    SetRole { role: Option<String> },
    SetKv { key: String, value: String },
    DeleteKv { key: String },
    GetKv { key: String },
    CreateSchema(CreateSchema),
    DropSchema(DropSchema),
    CreateDatabase(CreateDatabase),
    DropDatabase(DropDatabase),
    RenameDatabase(RenameDatabase),
    CreateTablespace(CreateTablespace),
    DropTablespace(DropTablespace),
    RenameTablespace(RenameTablespace),
    CreateTable(CreateTable),
    AddPrimaryKey(AddPrimaryKey),
    AddUniqueConstraint(AddUniqueConstraint),
    AddCheckConstraint(AddCheckConstraint),
    AddForeignKey(AddForeignKey),
    AddColumn(AddColumn),
    RenameTable(RenameTable),
    RenameColumn(RenameColumn),
    RenameConstraint(RenameConstraint),
    DropColumn(DropColumn),
    DropConstraint(DropConstraint),
    CreateIndex(CreateIndex),
    RenameIndex(RenameIndex),
    CreateView(CreateView),
    RenameView(RenameView),
    CreateMaterializedView(CreateMaterializedView),
    RefreshMaterializedView(RefreshMaterializedView),
    RenameMaterializedView(RenameMaterializedView),
    CreateFunction(CreateFunction),
    RenameFunction(RenameFunction),
    DropFunction(DropFunction),
    SelectFunction(SelectFunction),
    CreateExtension(CreateExtension),
    DropExtension(DropExtension),
    CreateSequence(CreateSequence),
    CreateDomain(CreateDomain),
    SequenceNextVal(SequenceNextVal),
    SequenceCurrVal(SequenceCurrVal),
    SequenceSetVal(SequenceSetVal),
    RenameSequence(RenameSequence),
    DropSequence(DropSequence),
    DropDomain(DropDomain),
    CreatePublication(CreatePublication),
    DropPublication(DropPublication),
    CreateSubscription(CreateSubscription),
    DropSubscription(DropSubscription),
    CreateRole(CreateRole),
    DropRole(DropRole),
    RenameRole(RenameRole),
    GrantTable(GrantTable),
    RevokeTable(RevokeTable),
    GrantDatabase(DatabasePrivileges),
    RevokeDatabase(DatabasePrivileges),
    GrantTablespace(TablespacePrivileges),
    RevokeTablespace(TablespacePrivileges),
    GrantFunction(FunctionPrivileges),
    RevokeFunction(FunctionPrivileges),
    GrantSchema(SchemaPrivileges),
    RevokeSchema(SchemaPrivileges),
    GrantDefaultTablePrivileges(DefaultTablePrivileges),
    RevokeDefaultTablePrivileges(DefaultTablePrivileges),
    DropTable(DropTable),
    TruncateTable(TruncateTable),
    DropIndex(DropIndex),
    DropMaterializedView(DropMaterializedView),
    DropView(DropView),
    AlterColumnDefault(AlterColumnDefault),
    CommentOn(CommentOn),
    Insert(Insert),
    Delete(Delete),
    Update(Update),
    Select(Select),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSchema {
    pub name: String,
    pub if_not_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropSchema {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateDatabase {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropDatabase {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameDatabase {
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTablespace {
    pub name: String,
    pub location: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTablespace {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameTablespace {
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTable {
    pub table: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Option<PrimaryKey>,
    pub unique_constraints: Vec<UniqueConstraint>,
    pub check_constraints: Vec<CheckConstraint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddPrimaryKey {
    pub table: String,
    pub name: String,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND key has len > 1.
    pub columns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimaryKey {
    pub name: Option<String>,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND primary key has len > 1.
    pub columns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniqueConstraint {
    pub name: Option<String>,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND unique constraint has len > 1.
    pub columns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddUniqueConstraint {
    pub table: String,
    pub name: String,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND unique constraint has len > 1.
    pub columns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckConstraint {
    pub name: Option<String>,
    pub filter: SelectFilter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddCheckConstraint {
    pub table: String,
    pub name: String,
    pub filter: SelectFilter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddForeignKey {
    pub table: String,
    pub name: String,
    pub column: String,
    pub referenced_table: String,
    pub referenced_column: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddColumn {
    pub table: String,
    pub column: ColumnDef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropColumn {
    pub table: String,
    pub column: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameTable {
    pub old_name: String,
    pub new_name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameColumn {
    pub table: String,
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameConstraint {
    pub table: String,
    pub old_name: String,
    pub new_name: String,
    pub table_if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropConstraint {
    pub table: String,
    pub name: String,
    pub table_if_exists: bool,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateIndex {
    pub name: String,
    pub table: String,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND index/constraint has len > 1.
    pub columns: Vec<String>,
    pub unique: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameIndex {
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateView {
    pub name: String,
    pub query: Select,
    pub definition: String,
    pub or_replace: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameView {
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateMaterializedView {
    pub name: String,
    pub query: Select,
    pub definition: String,
    pub with_data: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefreshMaterializedView {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameMaterializedView {
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateFunction {
    pub name: String,
    pub return_type: SqlType,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameFunction {
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropFunction {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectFunction {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSequence {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceNextVal {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceCurrVal {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceSetVal {
    pub name: String,
    pub value: i64,
    pub is_called: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameSequence {
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropSequence {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationTarget {
    AllTables,
    Tables(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreatePublication {
    pub name: String,
    pub target: PublicationTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropPublication {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSubscription {
    pub name: String,
    pub connection: String,
    pub publications: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropSubscription {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateRole {
    pub name: String,
    pub login: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropRole {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameRole {
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTable {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncateTable {
    pub name: String,
    pub restart_identity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropIndex {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropMaterializedView {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropView {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlterColumnDefault {
    pub table: String,
    pub column: String,
    pub default: Option<ColumnDefault>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentOn {
    pub target: CommentTarget,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommentTarget {
    Database { database: String },
    Role { role: String },
    Schema { schema: String },
    Tablespace { tablespace: String },
    Table { table: String },
    Column { table: String, column: String },
    Index { index: String },
    View { view: String },
    MaterializedView { materialized_view: String },
    Extension { extension: String },
    Function { function: String },
    Sequence { sequence: String },
    Domain { domain: String },
    Publication { publication: String },
    Subscription { subscription: String },
    Constraint { table: String, constraint: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub ty: SqlType,
    pub domain: Option<String>,
    pub default: Option<ColumnDefault>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateDomain {
    pub name: String,
    pub base_type: SqlType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropDomain {
    pub domains: Vec<String>,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateExtension {
    pub name: String,
    pub if_not_exists: bool,
    pub schema: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropExtension {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnDefault {
    Literal(SqlValue),
    SequenceNextVal {
        sequence: String,
        create_if_missing: bool,
    },
}

/// A storable column type. `Numeric` carries the PostgreSQL `numeric(p,s)` typmod
/// (`precision`/`scale`); every variant's payload is `Copy`, so `SqlType` stays
/// `Copy` exactly like the original `Int4`/`Text`-only enum (the ~60 `== SqlType::Int4`
/// guards and by-value passes are unaffected by the widening).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlType {
    /// PostgreSQL `smallint` (int2) — stored widened to i32 in the int4 device section (so it reuses
    /// the int4 compare path); arithmetic (int16-bounds overflow) is a follow-on.
    Int2,
    Int4,
    Int8,
    /// PostgreSQL `numeric(precision, scale)` — stored as a fixed-point [`Decimal128`].
    Numeric {
        precision: u8,
        scale: u8,
    },
    Bool,
    Text,
    /// PostgreSQL `date` — stored as i32 DAYS since 2000-01-01 (so it reuses the int4 device path).
    Date,
    /// PostgreSQL `timestamp` (without time zone) — stored as i64 MICROSECONDS since 2000-01-01
    /// 00:00:00 (so it reuses the int8 device path).
    Timestamp,
    /// PostgreSQL `uuid` — 16 raw bytes, compared byte-wise (unsigned). Reuses the i128 (16-byte)
    /// residency section, but with a byte-wise compare kernel rather than the signed i128 one.
    Uuid,
}

/// The default `numeric` typmod when a `NUMERIC`/`DECIMAL` column omits `(p,s)`.
/// PostgreSQL treats unconstrained `numeric` specially; we pin a wide fixed typmod
/// (38 significant digits — the i128 mantissa ceiling) so values round-trip without
/// a bignum fallback (>38 digits is a documented, errored edge — a future milestone).
pub const NUMERIC_DEFAULT_PRECISION: u8 = 38;
pub const NUMERIC_DEFAULT_SCALE: u8 = 0;

pub const SUPPORTED_SQL_TYPES: [SqlType; 9] = [
    SqlType::Int2,
    SqlType::Int4,
    SqlType::Int8,
    SqlType::Numeric {
        precision: NUMERIC_DEFAULT_PRECISION,
        scale: NUMERIC_DEFAULT_SCALE,
    },
    SqlType::Bool,
    SqlType::Text,
    SqlType::Date,
    SqlType::Timestamp,
    SqlType::Uuid,
];

impl SqlType {
    pub const fn postgres_oid(self) -> u32 {
        match self {
            Self::Int2 => 21,
            Self::Int4 => 23,
            Self::Int8 => 20,
            Self::Numeric { .. } => 1700,
            Self::Bool => 16,
            Self::Text => 25,
            Self::Date => 1082,
            Self::Timestamp => 1114,
            Self::Uuid => 2950,
        }
    }

    pub const fn type_size(self) -> i16 {
        match self {
            Self::Int2 => 2,
            Self::Int4 => 4,
            Self::Int8 => 8,
            Self::Numeric { .. } => -1,
            Self::Bool => 1,
            Self::Text => -1,
            Self::Date => 4,
            Self::Timestamp => 8,
            Self::Uuid => 16,
        }
    }

    pub const fn catalog_name(self) -> &'static str {
        match self {
            Self::Int2 => "int2",
            Self::Int4 => "int4",
            Self::Int8 => "int8",
            Self::Numeric { .. } => "numeric",
            Self::Bool => "bool",
            Self::Text => "text",
            Self::Date => "date",
            Self::Timestamp => "timestamp",
            Self::Uuid => "uuid",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Insert {
    pub table: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<SqlValue>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delete {
    pub table: String,
    pub filter: Option<SelectFilter>,
    pub filters: Vec<SelectFilter>,
    pub filter_groups: Vec<Vec<SelectFilter>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Update {
    pub table: String,
    pub assignments: Vec<UpdateAssignment>,
    pub filter: Option<SelectFilter>,
    pub filters: Vec<SelectFilter>,
    pub filter_groups: Vec<Vec<SelectFilter>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateAssignment {
    pub column: String,
    pub value: SqlValue,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SqlValue {
    /// SQL `NULL` — the typeless absence of a value. Placed first so the *derived*
    /// `Ord`/`Eq` (used only for internal value-index keys and dedup, never for SQL
    /// truth) gives a deterministic total order with `Null` sorting lowest. SQL
    /// semantics (where `NULL = NULL` is UNKNOWN and a comparison to NULL is never
    /// TRUE) route through `compare_sql_values`/`select_filter_matches`, never the
    /// derived traits. See `docs/architecture/21-null-representation-and-three-valued-logic.md`.
    Null,
    Int4(i32),
    Int8(i64),
    /// A fixed-point decimal (`numeric`). Equality and ordering are scale-aligned via
    /// [`Decimal128`], so `1.0` and `1.00` compare equal — which is why a value-index
    /// equality lookup on a NUMERIC column must key on a canonical scale.
    Numeric(Decimal128),
    Bool(bool),
    Text(String),
    /// A `date` as i32 DAYS since 2000-01-01 (PostgreSQL's date epoch). Ordering is the natural
    /// integer ordering of the day count.
    Date(i32),
    /// A `timestamp` as i64 MICROSECONDS since 2000-01-01 00:00:00. Ordering is the natural integer
    /// ordering of the microsecond count.
    Timestamp(i64),
    /// A `uuid` as its 16 raw bytes. Ordering is the byte-wise (unsigned) order of `[u8; 16]`, which
    /// is how PostgreSQL compares uuids.
    Uuid([u8; 16]),
    /// A `smallint` (int2) as i16. Widens to int4 for comparison (PG's numeric tower).
    Int2(i16),
}

/// Parse a PostgreSQL boolean *value* literal (for a `bool` column / `::bool` cast).
/// Accepts the canonical wire forms plus the spelled-out / single-letter aliases
/// PostgreSQL recognizes, case-insensitively. Distinct from [`parse_bool_literal`],
/// which is the stricter `true`/`t`/`false`/`f`-only parser for option arguments.
fn parse_bool_value(input: &str) -> Option<bool> {
    let trimmed = input.trim();
    if trimmed.eq_ignore_ascii_case("t")
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
        || trimmed.eq_ignore_ascii_case("on")
        || trimmed == "1"
    {
        Some(true)
    } else if trimmed.eq_ignore_ascii_case("f")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed == "0"
    {
        Some(false)
    } else {
        None
    }
}

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

fn parse_transaction_chain_suffix(tokens: &[&str]) -> Option<bool> {
    if tokens.is_empty() {
        return Some(false);
    }

    if matches!(
        tokens,
        [and, chain]
            if and.eq_ignore_ascii_case("AND") && chain.eq_ignore_ascii_case("CHAIN")
    ) {
        return Some(true);
    }

    if matches!(
        tokens,
        [and, no, chain]
            if and.eq_ignore_ascii_case("AND")
                && no.eq_ignore_ascii_case("NO")
                && chain.eq_ignore_ascii_case("CHAIN")
    ) {
        return Some(false);
    }

    None
}

fn parse_transaction_control_chain(input: &str, keyword: &str) -> Option<bool> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, mut rest) = tokens.split_first()?;
    if !first.eq_ignore_ascii_case(keyword) {
        return None;
    }

    if let Some((scope, tail)) = rest.split_first() {
        if scope.eq_ignore_ascii_case("TRANSACTION") || scope.eq_ignore_ascii_case("WORK") {
            rest = tail;
        }
    }

    parse_transaction_chain_suffix(rest)
}

fn parse_flush_command(input: &str) -> Option<Command> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, rest) = tokens.split_first()?;
    if first.eq_ignore_ascii_case("CHECKPOINT") {
        return match rest {
            [] => Some(Command::Flush),
            _ => None,
        };
    }

    if !first.eq_ignore_ascii_case("FLUSH") {
        return None;
    }

    match rest {
        [] => Some(Command::Flush),
        [target] if target.eq_ignore_ascii_case("WAL") || target.eq_ignore_ascii_case("LOG") => {
            Some(Command::Flush)
        }
        [write_ahead]
            if write_ahead.eq_ignore_ascii_case("WRITE-AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITEAHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD_LOG")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD_WAL") =>
        {
            Some(Command::Flush)
        }
        [write_ahead, target]
            if (write_ahead.eq_ignore_ascii_case("WRITE-AHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITEAHEAD")
                || write_ahead.eq_ignore_ascii_case("WRITE_AHEAD"))
                && (target.eq_ignore_ascii_case("LOG") || target.eq_ignore_ascii_case("WAL")) =>
        {
            Some(Command::Flush)
        }
        [write, ahead]
            if write.eq_ignore_ascii_case("WRITE") && ahead.eq_ignore_ascii_case("AHEAD") =>
        {
            Some(Command::Flush)
        }
        [write, ahead, target]
            if write.eq_ignore_ascii_case("WRITE")
                && ahead.eq_ignore_ascii_case("AHEAD")
                && (target.eq_ignore_ascii_case("LOG") || target.eq_ignore_ascii_case("WAL")) =>
        {
            Some(Command::Flush)
        }
        _ => None,
    }
}

fn parse_reset_command(input: &str) -> Option<Result<Command, ParseError>> {
    let tokens: Vec<_> = input.split_whitespace().collect();
    let (first, rest) = tokens.split_first()?;

    if first.eq_ignore_ascii_case("RESET") {
        return Some(match rest {
            [target]
                if target.eq_ignore_ascii_case("ALL")
                    || target.eq_ignore_ascii_case("ROLE")
                    || target.eq_ignore_ascii_case("AUTHORIZATION")
                    || target.eq_ignore_ascii_case("AUTH") =>
            {
                Ok(Command::ResetAll)
            }
            [scope, role]
                if (scope.eq_ignore_ascii_case("SESSION")
                    || scope.eq_ignore_ascii_case("LOCAL"))
                    && role.eq_ignore_ascii_case("ROLE") =>
            {
                Ok(Command::ResetAll)
            }
            [session, authorization]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH")) =>
            {
                Ok(Command::ResetAll)
            }
            [session, authorization, default]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH"))
                    && default.eq_ignore_ascii_case("DEFAULT") =>
            {
                Ok(Command::ResetAll)
            }
            [session, authorization, to, default]
                if session.eq_ignore_ascii_case("SESSION")
                    && (authorization.eq_ignore_ascii_case("AUTHORIZATION")
                        || authorization.eq_ignore_ascii_case("AUTH"))
                    && to.eq_ignore_ascii_case("TO")
                    && default.eq_ignore_ascii_case("DEFAULT") =>
            {
                Ok(Command::ResetAll)
            }
            _ => Err(ParseError::InvalidReset),
        });
    }

    if first.eq_ignore_ascii_case("DISCARD") {
        return Some(match rest {
            [target]
                if target.eq_ignore_ascii_case("ALL")
                    || target.eq_ignore_ascii_case("TEMP")
                    || target.eq_ignore_ascii_case("TEMPORARY")
                    || target.eq_ignore_ascii_case("PLANS")
                    || target.eq_ignore_ascii_case("SEQUENCES") =>
            {
                Ok(Command::ResetAll)
            }
            [scope, kind]
                if (scope.eq_ignore_ascii_case("TEMP")
                    || scope.eq_ignore_ascii_case("TEMPORARY"))
                    && (kind.eq_ignore_ascii_case("TABLE")
                        || kind.eq_ignore_ascii_case("TABLES")) =>
            {
                Ok(Command::ResetAll)
            }
            _ => Err(ParseError::InvalidReset),
        });
    }

    if first.eq_ignore_ascii_case("DEALLOCATE") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "DEALLOCATE") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim_start();
        if rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }

        if let Some(after_prepared) = strip_keyword_prefix_case_insensitive(rest, "PREPARED") {
            let tail = after_prepared.trim_start();
            return Some(
                if parse_reset_identifier(tail).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidReset)
                },
            );
        }

        if let Some(after_prepare) = strip_keyword_prefix_case_insensitive(rest, "PREPARE") {
            let tail = after_prepare.trim_start();
            return Some(
                if parse_reset_identifier(tail).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidReset)
                },
            );
        }

        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("CLOSE") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "CLOSE") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim_start();
        if rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }
        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("UNLISTEN") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "UNLISTEN") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let rest = rest.trim();
        if rest.is_empty() || rest == "*" || rest.eq_ignore_ascii_case("ALL") {
            return Some(Ok(Command::ResetAll));
        }
        return Some(
            if parse_reset_identifier(rest).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("LISTEN") {
        let Some(rest) = strip_keyword_prefix_case_insensitive(input, "LISTEN") else {
            return Some(Err(ParseError::InvalidReset));
        };
        return Some(
            if parse_reset_identifier(rest.trim()).is_some_and(|(_, tail)| tail.trim().is_empty()) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    if first.eq_ignore_ascii_case("NOTIFY") {
        let Some(raw_rest) = strip_keyword_prefix_case_insensitive(input, "NOTIFY") else {
            return Some(Err(ParseError::InvalidReset));
        };
        let Some((_, tail_after_channel)) = parse_reset_identifier(raw_rest.trim_start()) else {
            return Some(Err(ParseError::InvalidReset));
        };
        let tail_after_channel = tail_after_channel.trim_start();
        if tail_after_channel.is_empty() {
            return Some(Ok(Command::ResetAll));
        }
        let Some(payload) = tail_after_channel.strip_prefix(',') else {
            return Some(Err(ParseError::InvalidReset));
        };
        let payload = payload.trim_start();
        return Some(
            if !payload.starts_with(',') && notify_payload_fragment_is_non_empty(payload) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidReset)
            },
        );
    }

    None
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

fn parse_reset_identifier(input: &str) -> Option<(&str, &str)> {
    let s = input.trim_start();
    if s.is_empty() {
        return None;
    }

    if s.starts_with('"') {
        let bytes = s.as_bytes();
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                if i == 1 {
                    return None;
                }
                let end = i + 1;
                return Some((&s[..end], &s[end..]));
            }
            i += 1;
        }
        return None;
    }

    let mut chars = s.char_indices();
    let (_, first) = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return None;
    }

    let mut end = first.len_utf8();
    for (idx, ch) in chars {
        if ch.is_whitespace() || ch == ',' {
            end = idx;
            break;
        }
        if !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()) {
            return None;
        }
        end = idx + ch.len_utf8();
    }

    Some((&s[..end], &s[end..]))
}

fn notify_payload_fragment_is_non_empty(fragment: &str) -> bool {
    let trimmed = fragment.trim();
    if trimmed.is_empty() || trimmed.trim_matches(',').trim().is_empty() {
        return false;
    }

    let mut in_single_quote = false;
    let mut single_quote_backslash_escapes = false;
    let mut in_double_quote = false;
    let chars: Vec<char> = trimmed.chars().collect();
    let mut idx = 0;

    while idx < chars.len() {
        let ch = chars[idx];
        if in_single_quote {
            if single_quote_backslash_escapes && ch == '\\' && idx + 1 < chars.len() {
                idx += 2;
                continue;
            }

            if ch == '\'' {
                if idx + 1 < chars.len() && chars[idx + 1] == '\'' {
                    idx += 2;
                    continue;
                }
                in_single_quote = false;
                single_quote_backslash_escapes = false;
            }
            idx += 1;
            continue;
        }

        if in_double_quote {
            if ch == '"' {
                if idx + 1 < chars.len() && chars[idx + 1] == '"' {
                    idx += 2;
                    continue;
                }
                in_double_quote = false;
            }
            idx += 1;
            continue;
        }

        if ch == '$' {
            if let Some((delim, after_start)) = parse_notify_dollar_quote_start(&chars, idx) {
                let mut scan = after_start;
                let mut found = false;
                while scan + delim.len() <= chars.len() {
                    if chars[scan..scan + delim.len()] == delim[..] {
                        idx = scan + delim.len();
                        found = true;
                        break;
                    }
                    scan += 1;
                }
                if !found {
                    return false;
                }
                continue;
            }
        }

        if ch.is_whitespace() {
            if chars[idx + 1..].iter().any(|c| !c.is_whitespace()) {
                return false;
            }
            break;
        }

        if ch == '\'' {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 1;
            continue;
        }

        if matches!(ch, 'e' | 'E') && chars.get(idx + 1) == Some(&'\'') {
            in_single_quote = true;
            single_quote_backslash_escapes = true;
            idx += 2;
            continue;
        }

        if matches!(ch, 'b' | 'B' | 'x' | 'X') && chars.get(idx + 1) == Some(&'\'') {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 2;
            continue;
        }

        if matches!(ch, 'u' | 'U')
            && chars.get(idx + 1) == Some(&'&')
            && chars.get(idx + 2) == Some(&'\'')
        {
            in_single_quote = true;
            single_quote_backslash_escapes = false;
            idx += 3;
            continue;
        }

        match ch {
            '"' => in_double_quote = true,
            ',' => return false,
            _ => {}
        }
        idx += 1;
    }

    !(in_single_quote || in_double_quote)
}

fn parse_notify_dollar_quote_start(chars: &[char], start: usize) -> Option<(Vec<char>, usize)> {
    if chars.get(start) != Some(&'$') {
        return None;
    }
    let mut idx = start + 1;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch == '$' {
            return Some((chars[start..=idx].to_vec(), idx + 1));
        }
        if !(ch.is_ascii_alphanumeric() || ch == '_') {
            return None;
        }
        idx += 1;
    }
    None
}

fn split_set_key_value(rest: &str) -> Option<(&str, &str)> {
    if let Some((k, v)) = rest.split_once('=') {
        return Some((k, v));
    }

    let trimmed = rest.trim();
    let (key, tail) = trimmed.split_once(char::is_whitespace)?;
    let tail = tail.trim_start();
    if tail.len() < 2 {
        return None;
    }

    let (keyword, remainder) = tail.split_at(2);
    if !keyword.eq_ignore_ascii_case("TO") {
        return None;
    }

    if remainder.is_empty() || !remainder.starts_with(char::is_whitespace) {
        return None;
    }

    Some((key, remainder.trim_start()))
}

fn strip_set_scope_prefix<'a>(input: &'a str, scope: &str) -> Option<&'a str> {
    let trimmed = input.trim_start();
    let after_scope = strip_keyword_prefix_case_insensitive(trimmed, scope)?;
    if after_scope.trim().is_empty() {
        return None;
    }
    Some(after_scope.trim_start())
}

fn parse_relational_command(
    input: &str,
    allow_catalog_schemas: bool,
) -> Option<Result<Command, ParseError>> {
    let first = input.split_whitespace().next()?;
    if first.eq_ignore_ascii_case("CREATE") {
        let second = input.split_whitespace().nth(1)?;
        if second.eq_ignore_ascii_case("TABLE") {
            return Some(parse_create_table(input).map(Command::CreateTable));
        }
        if second.eq_ignore_ascii_case("SCHEMA") {
            return Some(parse_create_schema(input).map(Command::CreateSchema));
        }
        if second.eq_ignore_ascii_case("DATABASE") {
            return Some(parse_create_database(input).map(Command::CreateDatabase));
        }
        if second.eq_ignore_ascii_case("TABLESPACE") {
            return Some(parse_create_tablespace(input).map(Command::CreateTablespace));
        }
        if second.eq_ignore_ascii_case("INDEX") || second.eq_ignore_ascii_case("UNIQUE") {
            return Some(parse_create_index(input).map(Command::CreateIndex));
        }
        if second.eq_ignore_ascii_case("VIEW") {
            return Some(parse_create_view(input).map(Command::CreateView));
        }
        if second.eq_ignore_ascii_case("MATERIALIZED") {
            let third = input.split_whitespace().nth(2)?;
            if third.eq_ignore_ascii_case("VIEW") {
                return Some(
                    parse_create_materialized_view(input).map(Command::CreateMaterializedView),
                );
            }
        }
        if second.eq_ignore_ascii_case("FUNCTION") {
            return Some(parse_create_function(input).map(Command::CreateFunction));
        }
        if second.eq_ignore_ascii_case("SEQUENCE") {
            return Some(parse_create_sequence(input).map(Command::CreateSequence));
        }
        if second.eq_ignore_ascii_case("DOMAIN") {
            return Some(parse_create_domain(input).map(Command::CreateDomain));
        }
        if second.eq_ignore_ascii_case("PUBLICATION") {
            return Some(parse_create_publication(input).map(Command::CreatePublication));
        }
        if second.eq_ignore_ascii_case("SUBSCRIPTION") {
            return Some(parse_create_subscription(input).map(Command::CreateSubscription));
        }
        if second.eq_ignore_ascii_case("EXTENSION") {
            return Some(parse_create_extension(input).map(Command::CreateExtension));
        }
        if second.eq_ignore_ascii_case("ROLE") || second.eq_ignore_ascii_case("USER") {
            return Some(parse_create_role(input).map(Command::CreateRole));
        }
        if second.eq_ignore_ascii_case("OR") {
            let third = input.split_whitespace().nth(2)?;
            let fourth = input.split_whitespace().nth(3)?;
            if third.eq_ignore_ascii_case("REPLACE") && fourth.eq_ignore_ascii_case("VIEW") {
                return Some(parse_create_view(input).map(Command::CreateView));
            }
        }
        return Some(Err(ParseError::InvalidRelationalSql));
    }
    if first.eq_ignore_ascii_case("DROP") {
        let second = input.split_whitespace().nth(1)?;
        if second.eq_ignore_ascii_case("TABLE") {
            return Some(parse_drop_table(input).map(Command::DropTable));
        }
        if second.eq_ignore_ascii_case("SCHEMA") {
            return Some(parse_drop_schema(input).map(Command::DropSchema));
        }
        if second.eq_ignore_ascii_case("DATABASE") {
            return Some(parse_drop_database(input).map(Command::DropDatabase));
        }
        if second.eq_ignore_ascii_case("TABLESPACE") {
            return Some(parse_drop_tablespace(input).map(Command::DropTablespace));
        }
        if second.eq_ignore_ascii_case("INDEX") {
            return Some(parse_drop_index(input).map(Command::DropIndex));
        }
        if second.eq_ignore_ascii_case("VIEW") {
            return Some(parse_drop_view(input).map(Command::DropView));
        }
        if second.eq_ignore_ascii_case("MATERIALIZED") {
            let third = input.split_whitespace().nth(2)?;
            if third.eq_ignore_ascii_case("VIEW") {
                return Some(
                    parse_drop_materialized_view(input).map(Command::DropMaterializedView),
                );
            }
        }
        if second.eq_ignore_ascii_case("FUNCTION") {
            return Some(parse_drop_function(input).map(Command::DropFunction));
        }
        if second.eq_ignore_ascii_case("SEQUENCE") {
            return Some(parse_drop_sequence(input).map(Command::DropSequence));
        }
        if second.eq_ignore_ascii_case("DOMAIN") {
            return Some(parse_drop_domain(input).map(Command::DropDomain));
        }
        if second.eq_ignore_ascii_case("EXTENSION") {
            return Some(parse_drop_extension(input).map(Command::DropExtension));
        }
        if second.eq_ignore_ascii_case("PUBLICATION") {
            return Some(parse_drop_publication(input).map(Command::DropPublication));
        }
        if second.eq_ignore_ascii_case("SUBSCRIPTION") {
            return Some(parse_drop_subscription(input).map(Command::DropSubscription));
        }
        if second.eq_ignore_ascii_case("ROLE") || second.eq_ignore_ascii_case("USER") {
            return Some(parse_drop_role(input).map(Command::DropRole));
        }
        return Some(Err(ParseError::InvalidRelationalSql));
    }
    if first.eq_ignore_ascii_case("TRUNCATE") {
        return Some(parse_truncate_table(input).map(Command::TruncateTable));
    }
    if first.eq_ignore_ascii_case("REFRESH") {
        return Some(parse_refresh_materialized_view(input).map(Command::RefreshMaterializedView));
    }
    if let Some(command) = acl::parse_acl_command(input) {
        return Some(command);
    }
    if first.eq_ignore_ascii_case("ALTER") {
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("DATABASE"))
        {
            return Some(parse_rename_database(input).map(Command::RenameDatabase));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("TABLESPACE"))
        {
            return Some(parse_rename_tablespace(input).map(Command::RenameTablespace));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("ROLE"))
        {
            return Some(parse_rename_role(input).map(Command::RenameRole));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("INDEX"))
        {
            return Some(parse_rename_index(input).map(Command::RenameIndex));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("MATERIALIZED"))
        {
            return Some(
                parse_rename_materialized_view(input).map(Command::RenameMaterializedView),
            );
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("VIEW"))
        {
            return Some(parse_rename_view(input).map(Command::RenameView));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("FUNCTION"))
        {
            return Some(parse_rename_function(input).map(Command::RenameFunction));
        }
        if input
            .split_whitespace()
            .nth(1)
            .is_some_and(|second| second.eq_ignore_ascii_case("SEQUENCE"))
        {
            return Some(parse_rename_sequence(input).map(Command::RenameSequence));
        }
        if find_keyword_outside_quotes(input, "RENAME").is_some() {
            if parse_rename_constraint(input).is_ok() {
                return Some(parse_rename_constraint(input).map(Command::RenameConstraint));
            }
            if parse_rename_table(input).is_ok() {
                return Some(parse_rename_table(input).map(Command::RenameTable));
            }
            return Some(parse_rename_column(input).map(Command::RenameColumn));
        }
        if find_keyword_outside_quotes(input, "DROP").is_some()
            && find_keyword_outside_quotes(input, "DEFAULT").is_none()
        {
            if parse_drop_column(input).is_ok() {
                return Some(parse_drop_column(input).map(Command::DropColumn));
            }
            return Some(parse_drop_table_constraint(input).map(Command::DropConstraint));
        }
        if find_keyword_outside_quotes(input, "ADD").is_some() {
            return Some(parse_alter_table_add(input));
        }
        return Some(parse_alter_column_default(input).map(Command::AlterColumnDefault));
    }
    if first.eq_ignore_ascii_case("COMMENT") {
        return Some(parse_comment_on(input).map(Command::CommentOn));
    }
    if first.eq_ignore_ascii_case("INSERT") {
        return Some(parse_insert(input).map(Command::Insert));
    }
    if first.eq_ignore_ascii_case("UPDATE")
        && find_keyword_outside_quotes(input, "SET").is_some()
        && find_keyword_outside_quotes(input, "WHERE").is_some()
    {
        return Some(parse_update(input).map(Command::Update));
    }
    if first.eq_ignore_ascii_case("DELETE")
        && strip_keyword_prefix_case_insensitive(input, "DELETE")
            .map(str::trim_start)
            .and_then(|tail| strip_keyword_prefix_case_insensitive(tail, "FROM"))
            .is_some_and(|tail| find_keyword_outside_quotes(tail, "WHERE").is_some())
    {
        return Some(parse_delete(input).map(Command::Delete));
    }
    if first.eq_ignore_ascii_case("SELECT") {
        if let Ok(sequence_command) = parse_sequence_value_function(input) {
            return Some(Ok(sequence_command));
        }
        if let Ok(function_command) = parse_select_function(input) {
            return Some(Ok(function_command));
        }
        return Some(parse_select(input, allow_catalog_schemas).map(Command::Select));
    }
    None
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

fn parse_alter_column_default(input: &str) -> Result<AlterColumnDefault, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let alter_pos =
        find_keyword_outside_quotes(rest, "ALTER").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..alter_pos].trim())?;
    let rest = rest[alter_pos + "ALTER".len()..].trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "COLUMN")
        .map(str::trim_start)
        .unwrap_or(rest);
    if let Some(set_pos) = find_keyword_outside_quotes(rest, "SET") {
        let column = normalize_identifier(rest[..set_pos].trim())?;
        let rest = strip_keyword_prefix_case_insensitive(
            rest[set_pos + "SET".len()..].trim_start(),
            "DEFAULT",
        )
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
        if rest.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        return Ok(AlterColumnDefault {
            table,
            column,
            default: Some(parse_column_default_expr(rest, None)?),
        });
    }

    let drop_pos =
        find_keyword_outside_quotes(rest, "DROP").ok_or(ParseError::InvalidRelationalSql)?;
    let column = normalize_identifier(rest[..drop_pos].trim())?;
    let rest = strip_keyword_prefix_case_insensitive(
        rest[drop_pos + "DROP".len()..].trim_start(),
        "DEFAULT",
    )
    .ok_or(ParseError::InvalidRelationalSql)?
    .trim();
    if !rest.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(AlterColumnDefault {
        table,
        column,
        default: None,
    })
}

fn parse_add_primary_key(input: &str) -> Result<AddPrimaryKey, ParseError> {
    let (table, name, rest) = parse_alter_table_add_constraint(input)?;
    let rest = strip_keyword_prefix_case_insensitive(rest, "PRIMARY")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "KEY"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let columns = parse_constraint_columns(rest)?;
    Ok(AddPrimaryKey {
        table,
        name,
        column: columns[0].clone(),
        columns,
    })
}

fn parse_add_unique_constraint(input: &str) -> Result<AddUniqueConstraint, ParseError> {
    let (table, name, rest) = parse_alter_table_add_constraint(input)?;
    let rest = strip_keyword_prefix_case_insensitive(rest, "UNIQUE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let columns = parse_constraint_columns(rest)?;
    Ok(AddUniqueConstraint {
        table,
        name,
        column: columns[0].clone(),
        columns,
    })
}

fn parse_add_check_constraint(input: &str) -> Result<AddCheckConstraint, ParseError> {
    let (table, name, rest) = parse_alter_table_add_constraint(input)?;
    let filter = parse_check_constraint_filter(rest)?;
    Ok(AddCheckConstraint {
        table,
        name,
        filter,
    })
}

fn parse_add_foreign_key(input: &str) -> Result<AddForeignKey, ParseError> {
    let (table, name, rest) = parse_alter_table_add_constraint(input)?;
    let rest = strip_keyword_prefix_case_insensitive(rest, "FOREIGN")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "KEY"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(rest, open).ok_or(ParseError::InvalidRelationalSql)?;
    let columns = split_csv(&rest[open + 1..close])?;
    let [column] = columns.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let column = normalize_identifier(column.trim())?;
    let rest = rest[close + 1..].trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "REFERENCES")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(rest, open).ok_or(ParseError::InvalidRelationalSql)?;
    if close <= open || !rest[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let referenced_table = normalize_relation_identifier(rest[..open].trim())?;
    let referenced_columns = split_csv(&rest[open + 1..close])?;
    let [referenced_column] = referenced_columns.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(AddForeignKey {
        table,
        name,
        column,
        referenced_table,
        referenced_column: normalize_identifier(referenced_column.trim())?,
    })
}

fn parse_add_table_constraint(input: &str) -> Result<Command, ParseError> {
    let (_, _, rest) = parse_alter_table_add_constraint(input)?;
    if strip_keyword_prefix_case_insensitive(rest, "PRIMARY").is_some() {
        return parse_add_primary_key(input).map(Command::AddPrimaryKey);
    }
    if strip_keyword_prefix_case_insensitive(rest, "UNIQUE").is_some() {
        return parse_add_unique_constraint(input).map(Command::AddUniqueConstraint);
    }
    if strip_keyword_prefix_case_insensitive(rest, "CHECK").is_some() {
        return parse_add_check_constraint(input).map(Command::AddCheckConstraint);
    }
    if strip_keyword_prefix_case_insensitive(rest, "FOREIGN").is_some() {
        return parse_add_foreign_key(input).map(Command::AddForeignKey);
    }
    Err(ParseError::InvalidRelationalSql)
}

fn parse_alter_table_add(input: &str) -> Result<Command, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let add_pos =
        find_keyword_outside_quotes(rest, "ADD").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..add_pos].trim())?;
    let add_tail = rest[add_pos + "ADD".len()..].trim_start();
    if strip_keyword_prefix_case_insensitive(add_tail, "CONSTRAINT").is_some() {
        return parse_add_table_constraint(input);
    }
    let column_tail = strip_keyword_prefix_case_insensitive(add_tail, "COLUMN")
        .map(str::trim_start)
        .unwrap_or(add_tail);
    Ok(Command::AddColumn(AddColumn {
        table,
        column: parse_column_def(column_tail)?,
    }))
}

fn parse_drop_column(input: &str) -> Result<DropColumn, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let drop_pos =
        find_keyword_outside_quotes(rest, "DROP").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..drop_pos].trim())?;
    rest = rest[drop_pos + "DROP".len()..].trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "COLUMN")
        .map(str::trim_start)
        .unwrap_or(rest);
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let columns = split_csv(rest)?;
    let [column] = columns.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(DropColumn {
        table,
        column: normalize_identifier(column.trim())?,
    })
}

fn parse_rename_table(input: &str) -> Result<RenameTable, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let if_exists = if let Some(remaining) = strip_keyword_prefix_case_insensitive(rest, "IF")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "EXISTS"))
    {
        rest = remaining.trim_start();
        true
    } else {
        false
    };
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_relation_identifier(rest[..rename_pos].trim())?;
    rest = rest[rename_pos + "RENAME".len()..].trim_start();
    let new_tail = strip_keyword_prefix_case_insensitive(rest, "TO")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim();
    if new_tail.is_empty()
        || find_keyword_outside_quotes(new_tail, "CASCADE").is_some()
        || find_keyword_outside_quotes(new_tail, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameTable {
        old_name,
        new_name: normalize_identifier(new_tail)?,
        if_exists,
    })
}

fn parse_rename_column(input: &str) -> Result<RenameColumn, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..rename_pos].trim())?;
    rest = rest[rename_pos + "RENAME".len()..].trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "COLUMN")
        .map(str::trim_start)
        .unwrap_or(rest);
    let to_pos = find_keyword_outside_quotes(rest, "TO").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_identifier(rest[..to_pos].trim())?;
    let new_tail = rest[to_pos + "TO".len()..].trim();
    if new_tail.is_empty()
        || find_keyword_outside_quotes(new_tail, "CASCADE").is_some()
        || find_keyword_outside_quotes(new_tail, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameColumn {
        table,
        old_name,
        new_name: normalize_identifier(new_tail)?,
    })
}

fn parse_rename_constraint(input: &str) -> Result<RenameConstraint, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let table_if_exists = if let Some(remaining) = strip_keyword_prefix_case_insensitive(rest, "IF")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "EXISTS"))
    {
        rest = remaining.trim_start();
        true
    } else {
        false
    };
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let rename_pos =
        find_keyword_outside_quotes(rest, "RENAME").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..rename_pos].trim())?;
    rest = rest[rename_pos + "RENAME".len()..].trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "CONSTRAINT")
        .map(str::trim_start)
        .ok_or(ParseError::InvalidRelationalSql)?;
    let to_pos = find_keyword_outside_quotes(rest, "TO").ok_or(ParseError::InvalidRelationalSql)?;
    let old_name = normalize_identifier(rest[..to_pos].trim())?;
    let new_tail = rest[to_pos + "TO".len()..].trim();
    if new_tail.is_empty()
        || find_keyword_outside_quotes(new_tail, "CASCADE").is_some()
        || find_keyword_outside_quotes(new_tail, "RESTRICT").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(RenameConstraint {
        table,
        old_name,
        new_name: normalize_identifier(new_tail)?,
        table_if_exists,
    })
}

/// Split off the leading identifier (a column or type name) from `input`, returning
/// `(name, rest)`. The name runs to the first ASCII whitespace.
fn split_leading_word(input: &str) -> Option<(&str, &str)> {
    let input = input.trim_start();
    if input.is_empty() {
        return None;
    }
    let end = input.find(char::is_whitespace).unwrap_or(input.len());
    Some((&input[..end], input[end..].trim_start()))
}

/// Split a column type token from `input`, returning `(type_token, tail)`. The token
/// is a name optionally followed by a balanced `(...)` typmod group, so a spaced
/// `NUMERIC(12, 2)` is kept intact (unlike a naive whitespace split).
fn split_column_type(input: &str) -> Option<(&str, &str)> {
    let (word, rest) = split_leading_word(input)?;
    // A `(` may begin the word's typmod immediately, or follow after whitespace
    // (`NUMERIC (12,2)`); accept both, balancing parens within the original input.
    let after_word_offset = word.as_ptr() as usize - input.as_ptr() as usize + word.len();
    let rest_trimmed = rest.trim_start();
    if rest_trimmed.starts_with('(') {
        let open = after_word_offset + (rest.len() - rest_trimmed.len());
        let close = find_matching_paren(input, open)?;
        return Some((input[..close + 1].trim(), input[close + 1..].trim_start()));
    }
    Some((word, rest))
}

/// Resolve a column's declared type token to a `(SqlType, domain)` pair. A token that
/// is not a built-in type is treated as a domain reference (defaulting to `Int4`,
/// matching the pre-existing behavior). `serial`/`serial4` is handled by the caller.
fn resolve_column_type(token: &str) -> Result<(SqlType, Option<String>), ParseError> {
    match parse_supported_sql_type_name(token) {
        Some(ty) => Ok((ty, None)),
        None => Ok((SqlType::Int4, Some(normalize_relation_identifier(token)?))),
    }
}

fn parse_column_def(input: &str) -> Result<ColumnDef, ParseError> {
    let (name, after_name) = split_leading_word(input).ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_identifier(name)?;
    let (raw_ty, tail) = split_column_type(after_name).ok_or(ParseError::InvalidRelationalSql)?;
    let (ty, domain) = resolve_column_type(raw_ty)?;
    let tail = tail.to_string();
    if domain.is_some() && !tail.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let default = if tail.is_empty() {
        None
    } else {
        let default_value = strip_keyword_prefix_case_insensitive(&tail, "DEFAULT")
            .ok_or(ParseError::InvalidRelationalSql)?
            .trim();
        if default_value.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        Some(parse_typed_column_default(default_value, ty, None)?)
    };
    Ok(ColumnDef {
        name,
        ty,
        domain,
        default,
    })
}

fn parse_column_default_expr(
    input: &str,
    implicit_serial_sequence: Option<String>,
) -> Result<ColumnDefault, ParseError> {
    let trimmed = input.trim();
    if let Some(sequence) = implicit_serial_sequence {
        if !trimmed.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        return Ok(ColumnDefault::SequenceNextVal {
            sequence,
            create_if_missing: true,
        });
    }
    let function = trimmed
        .strip_prefix("pg_catalog.")
        .or_else(|| trimmed.strip_prefix("PG_CATALOG."))
        .unwrap_or(trimmed);
    if let Some(args) = function
        .strip_prefix("nextval")
        .or_else(|| function.strip_prefix("NEXTVAL"))
    {
        let args = args.trim_start();
        if !args.starts_with('(') {
            return Err(ParseError::InvalidRelationalSql);
        }
        let close = find_matching_paren(args, 0).ok_or(ParseError::InvalidRelationalSql)?;
        if !args[close + 1..].trim().is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        let parts = split_csv(&args[1..close])?;
        let [target] = parts.as_slice() else {
            return Err(ParseError::InvalidRelationalSql);
        };
        return Ok(ColumnDefault::SequenceNextVal {
            sequence: parse_sequence_regclass_arg(target.trim())?,
            create_if_missing: false,
        });
    }
    Ok(ColumnDefault::Literal(parse_sql_value(trimmed)?))
}

fn parse_typed_column_default(
    input: &str,
    ty: SqlType,
    implicit_serial_sequence: Option<String>,
) -> Result<ColumnDefault, ParseError> {
    let default = parse_column_default_expr(input, implicit_serial_sequence)?;
    match default {
        // A literal default is coerced to the column's declared type, so e.g. `DEFAULT 0`
        // on a NUMERIC column is stored as a `Numeric` (not the inferred `Int4`), and
        // `DEFAULT TRUE` on a BOOL column is a `Bool`. We re-parse from the rendered
        // literal text rather than trusting the inferred variant.
        // A NULL default is the typeless SQL null — it stays NULL for a column of ANY type, so it
        // skips the type re-coercion (which would otherwise round-trip it through its rendered text
        // "NULL" and mis-parse it as e.g. the text value 'NULL').
        ColumnDefault::Literal(SqlValue::Null) => Ok(ColumnDefault::Literal(SqlValue::Null)),
        ColumnDefault::Literal(value) => {
            let rendered = render_default_literal_for_coercion(&value);
            let coerced = parse_typed_value_from_str(&rendered, ty)?;
            Ok(ColumnDefault::Literal(coerced))
        }
        // `nextval(...)` (serial) is integer-only, as before.
        ColumnDefault::SequenceNextVal { .. } if ty == SqlType::Int4 => Ok(default),
        ColumnDefault::SequenceNextVal { .. } => Err(ParseError::InvalidRelationalSql),
    }
}

/// Render an inferred default literal back to the textual form `parse_typed_value_from_str`
/// expects, so it can be re-parsed at the column's declared type.
fn render_default_literal_for_coercion(value: &SqlValue) -> String {
    match value {
        SqlValue::Null => "NULL".to_string(),
        SqlValue::Int2(value) => value.to_string(),
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Int8(value) => value.to_string(),
        SqlValue::Numeric(value) => value.to_decimal_string(),
        SqlValue::Bool(value) => {
            if *value {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        SqlValue::Text(value) => value.clone(),
        SqlValue::Date(value) => crate::datetime::format_date(*value),
        SqlValue::Timestamp(value) => crate::datetime::format_timestamp(*value),
        SqlValue::Uuid(value) => crate::uuid::format_uuid(value),
    }
}

fn parse_drop_table_constraint(input: &str) -> Result<DropConstraint, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let table_if_exists = if let Some(after_if) = strip_keyword_prefix_case_insensitive(rest, "IF")
    {
        let after_exists = strip_keyword_prefix_case_insensitive(after_if.trim_start(), "EXISTS")
            .ok_or(ParseError::InvalidRelationalSql)?;
        rest = after_exists.trim_start();
        true
    } else {
        false
    };
    rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let drop_pos =
        find_keyword_outside_quotes(rest, "DROP").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..drop_pos].trim())?;
    rest = rest[drop_pos + "DROP".len()..].trim_start();
    rest = strip_keyword_prefix_case_insensitive(rest, "CONSTRAINT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let constraint_if_exists = if let Some(after_if) =
        strip_keyword_prefix_case_insensitive(rest, "IF")
    {
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
    let constraints = split_csv(rest)?;
    let [constraint] = constraints.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(DropConstraint {
        table,
        name: normalize_identifier(constraint.trim())?,
        table_if_exists,
        if_exists: constraint_if_exists,
    })
}

fn parse_alter_table_add_constraint(input: &str) -> Result<(String, String, &str), ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "ONLY")
        .map(str::trim_start)
        .unwrap_or(rest);
    let add_pos =
        find_keyword_outside_quotes(rest, "ADD").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..add_pos].trim())?;
    let rest = rest[add_pos + "ADD".len()..].trim_start();
    let rest = strip_keyword_prefix_case_insensitive(rest, "CONSTRAINT")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let primary_pos = find_keyword_outside_quotes(rest, "PRIMARY");
    let unique_pos = find_keyword_outside_quotes(rest, "UNIQUE");
    let check_pos = find_keyword_outside_quotes(rest, "CHECK");
    let foreign_pos = find_keyword_outside_quotes(rest, "FOREIGN");
    let constraint_pos = [primary_pos, unique_pos, check_pos, foreign_pos]
        .into_iter()
        .flatten()
        .min()
        .ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_identifier(rest[..constraint_pos].trim())?;
    Ok((table, name, rest[constraint_pos..].trim_start()))
}

fn parse_check_constraint_filter(rest: &str) -> Result<SelectFilter, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(rest, "CHECK")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if !rest.starts_with('(') {
        return Err(ParseError::InvalidRelationalSql);
    }
    let close = find_matching_paren(rest, 0).ok_or(ParseError::InvalidRelationalSql)?;
    if close != rest.len() - 1 {
        return Err(ParseError::InvalidRelationalSql);
    }
    let filter = parse_select_filter(&rest[1..close])?;
    if matches!(filter.op, SelectFilterOp::LikePrefix) {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(filter)
}

/// Parse a constraint column LIST `(a, b, ...)` — the compound-key form of
/// [`parse_single_constraint_column`]. Returns the ordered, normalized key columns (>= 1). A
/// COMPOUND PRIMARY KEY / UNIQUE constraint is `PRIMARY KEY (a, b)`; a single-column one is the
/// `[a]` special case (so callers get a uniform `Vec`). Rejects an empty list / trailing tokens.
fn parse_constraint_columns(rest: &str) -> Result<Vec<String>, ParseError> {
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(rest, open).ok_or(ParseError::InvalidRelationalSql)?;
    if close <= open || !rest[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let raw = split_csv(&rest[open + 1..close])?;
    if raw.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    raw.iter()
        .map(|column| normalize_identifier(column.trim()))
        .collect()
}

fn parse_create_schema(input: &str) -> Result<CreateSchema, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SCHEMA"))
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
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "AUTHORIZATION").is_some()
        || find_keyword_outside_quotes(rest, "CREATE").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(CreateSchema {
        name: normalize_identifier(rest)?,
        if_not_exists,
    })
}

fn parse_drop_schema(input: &str) -> Result<DropSchema, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "SCHEMA"))
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
    let schemas = split_csv(rest)?;
    let [schema] = schemas.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(DropSchema {
        name: normalize_identifier(schema.trim())?,
        if_exists,
    })
}

fn parse_create_table(input: &str) -> Result<CreateTable, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let open = rest.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = rest.rfind(')').ok_or(ParseError::InvalidRelationalSql)?;
    if close <= open || !rest[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let table = normalize_relation_identifier(rest[..open].trim())?;
    let mut columns = Vec::new();
    let mut primary_key = None;
    let mut unique_constraints = Vec::new();
    let mut check_constraints = Vec::new();
    for raw_column in split_csv(&rest[open + 1..close])? {
        let trimmed = raw_column.trim();
        if let Some(after_constraint) = strip_keyword_prefix_case_insensitive(trimmed, "CONSTRAINT")
        {
            let primary_pos = find_keyword_outside_quotes(after_constraint, "PRIMARY");
            let unique_pos = find_keyword_outside_quotes(after_constraint, "UNIQUE");
            let check_pos = find_keyword_outside_quotes(after_constraint, "CHECK");
            let constraint_pos = [primary_pos, unique_pos, check_pos]
                .into_iter()
                .flatten()
                .min()
                .ok_or(ParseError::InvalidRelationalSql)?;
            let name = normalize_identifier(after_constraint[..constraint_pos].trim())?;
            let rest = after_constraint[constraint_pos..].trim_start();
            if strip_keyword_prefix_case_insensitive(rest, "PRIMARY").is_some() {
                let rest = strip_keyword_prefix_case_insensitive(rest, "PRIMARY")
                    .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "KEY"))
                    .ok_or(ParseError::InvalidRelationalSql)?
                    .trim_start();
                let columns = parse_constraint_columns(rest)?;
                if primary_key.is_some() {
                    return Err(ParseError::InvalidRelationalSql);
                }
                primary_key = Some(PrimaryKey {
                    name: Some(name),
                    column: columns[0].clone(),
                    columns,
                });
            } else if let Some(rest) = strip_keyword_prefix_case_insensitive(rest, "UNIQUE") {
                let columns = parse_constraint_columns(rest.trim_start())?;
                unique_constraints.push(UniqueConstraint {
                    name: Some(name),
                    column: columns[0].clone(),
                    columns,
                });
            } else if strip_keyword_prefix_case_insensitive(rest, "CHECK").is_some() {
                check_constraints.push(CheckConstraint {
                    name: Some(name),
                    filter: parse_check_constraint_filter(rest)?,
                });
            } else {
                return Err(ParseError::InvalidRelationalSql);
            }
            continue;
        }
        if let Some(rest) = strip_keyword_prefix_case_insensitive(trimmed, "PRIMARY") {
            let rest = strip_keyword_prefix_case_insensitive(rest.trim_start(), "KEY")
                .ok_or(ParseError::InvalidRelationalSql)?
                .trim_start();
            let columns = parse_constraint_columns(rest)?;
            if primary_key.is_some() {
                return Err(ParseError::InvalidRelationalSql);
            }
            primary_key = Some(PrimaryKey {
                name: None,
                column: columns[0].clone(),
                columns,
            });
            continue;
        }
        if let Some(rest) = strip_keyword_prefix_case_insensitive(trimmed, "UNIQUE") {
            let columns = parse_constraint_columns(rest.trim_start())?;
            unique_constraints.push(UniqueConstraint {
                name: None,
                column: columns[0].clone(),
                columns,
            });
            continue;
        }
        if strip_keyword_prefix_case_insensitive(trimmed, "CHECK").is_some() {
            check_constraints.push(CheckConstraint {
                name: None,
                filter: parse_check_constraint_filter(trimmed)?,
            });
            continue;
        }
        let (name, after_name) =
            split_leading_word(raw_column).ok_or(ParseError::InvalidRelationalSql)?;
        let name = normalize_identifier(name)?;
        let (raw_ty, tail) =
            split_column_type(after_name).ok_or(ParseError::InvalidRelationalSql)?;
        let mut serial_sequence = None;
        let (ty, domain) =
            if raw_ty.eq_ignore_ascii_case("serial") || raw_ty.eq_ignore_ascii_case("serial4") {
                serial_sequence = Some(format!("{}_{}_seq", table, name));
                (SqlType::Int4, None)
            } else {
                resolve_column_type(raw_ty)?
            };
        let mut tail = tail.to_string();
        let mut column_primary_key = false;
        let mut column_unique = false;
        if let Some(primary_pos) = find_keyword_outside_quotes(&tail, "PRIMARY") {
            let after_primary =
                strip_keyword_prefix_case_insensitive(tail[primary_pos..].trim_start(), "PRIMARY")
                    .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "KEY"))
                    .ok_or(ParseError::InvalidRelationalSql)?;
            if !after_primary.trim().is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            tail = tail[..primary_pos].trim().to_string();
            column_primary_key = true;
        }
        if let Some(unique_pos) = find_keyword_outside_quotes(&tail, "UNIQUE") {
            let after_unique =
                strip_keyword_prefix_case_insensitive(tail[unique_pos..].trim_start(), "UNIQUE")
                    .ok_or(ParseError::InvalidRelationalSql)?;
            if !after_unique.trim().is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            tail = tail[..unique_pos].trim().to_string();
            column_unique = true;
        }
        let default = if tail.is_empty() && serial_sequence.is_none() {
            None
        } else {
            if domain.is_some() {
                return Err(ParseError::InvalidRelationalSql);
            }
            if serial_sequence.is_some() && !tail.is_empty() {
                return Err(ParseError::InvalidRelationalSql);
            }
            let default_value = if serial_sequence.is_some() {
                ""
            } else {
                strip_keyword_prefix_case_insensitive(&tail, "DEFAULT")
                    .ok_or(ParseError::InvalidRelationalSql)?
                    .trim()
            };
            if default_value.is_empty() && serial_sequence.is_none() {
                return Err(ParseError::InvalidRelationalSql);
            }
            Some(parse_typed_column_default(
                default_value,
                ty,
                serial_sequence,
            )?)
        };
        if column_primary_key {
            if primary_key.is_some() {
                return Err(ParseError::InvalidRelationalSql);
            }
            primary_key = Some(PrimaryKey {
                name: None,
                column: name.clone(),
                columns: vec![name.clone()],
            });
        }
        if column_unique {
            unique_constraints.push(UniqueConstraint {
                name: None,
                column: name.clone(),
                columns: vec![name.clone()],
            });
        }
        columns.push(ColumnDef {
            name,
            ty,
            domain,
            default,
        });
    }
    if columns.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if let Some(key) = &primary_key {
        if !columns.iter().any(|column| column.name == key.column) {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
    for unique in &unique_constraints {
        if !columns.iter().any(|column| column.name == unique.column) {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
    for check in &check_constraints {
        if !columns
            .iter()
            .any(|column| column.name == check.filter.column)
        {
            return Err(ParseError::InvalidRelationalSql);
        }
    }
    Ok(CreateTable {
        table,
        columns,
        primary_key,
        unique_constraints,
        check_constraints,
    })
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

fn parse_create_index(input: &str) -> Result<CreateIndex, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "CREATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let (unique, rest) =
        if let Some(after_unique) = strip_keyword_prefix_case_insensitive(rest, "UNIQUE") {
            (true, after_unique.trim_start())
        } else {
            (false, rest)
        };
    let rest = strip_keyword_prefix_case_insensitive(rest, "INDEX")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let on_pos = find_keyword_outside_quotes(rest, "ON").ok_or(ParseError::InvalidRelationalSql)?;
    let name = normalize_relation_identifier(rest[..on_pos].trim())?;
    let target = rest[on_pos + "ON".len()..].trim_start();
    let open = target.find('(').ok_or(ParseError::InvalidRelationalSql)?;
    let close = find_matching_paren(target, open).ok_or(ParseError::InvalidRelationalSql)?;
    if close <= open || !target[close + 1..].trim().is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let table_target = target[..open].trim();
    let table_target =
        if let Some((table, method)) = split_optional_create_index_method(table_target) {
            if !method.eq_ignore_ascii_case("btree") {
                return Err(ParseError::InvalidRelationalSql);
            }
            table
        } else {
            table_target
        };
    let table = normalize_relation_identifier(table_target)?;
    // The literal `CREATE INDEX ... (cols)` text form stays SINGLE-column for now (compound secondary
    // indexes are out of scope); a multi-column list is a clean parse error. The compound-PK path builds
    // `CreateIndex` internally (apply_add_primary_key) and carries `columns` there.
    let raw = split_csv(&target[open + 1..close])?;
    let [column] = raw.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    let column = normalize_identifier(column.trim())?;
    Ok(CreateIndex {
        name,
        table,
        column: column.clone(),
        columns: vec![column],
        unique,
    })
}

fn parse_drop_index(input: &str) -> Result<DropIndex, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "INDEX"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if strip_keyword_prefix_case_insensitive(rest, "CONCURRENTLY").is_some() {
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
    let indexes = split_csv(rest)?;
    if indexes.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    Ok(DropIndex {
        names: indexes
            .into_iter()
            .map(|index| normalize_relation_identifier(index.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_rename_index(input: &str) -> Result<RenameIndex, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "ALTER")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "INDEX"))
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
    Ok(RenameIndex {
        old_name,
        new_name: normalize_identifier(after_to)?,
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

fn parse_drop_table(input: &str) -> Result<DropTable, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "DROP")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "TABLE"))
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
    if rest.is_empty() || find_keyword_outside_quotes(rest, "CASCADE").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    if find_keyword_outside_quotes(rest, "RESTRICT").is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let tables = split_csv(rest)?;
    if tables.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(DropTable {
        names: tables
            .into_iter()
            .map(|table| normalize_relation_identifier(table.trim()))
            .collect::<Result<Vec<_>, _>>()?,
        if_exists,
    })
}

fn parse_truncate_table(input: &str) -> Result<TruncateTable, ParseError> {
    let mut rest = strip_keyword_prefix_case_insensitive(input, "TRUNCATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    if let Some(after_table) = strip_keyword_prefix_case_insensitive(rest, "TABLE") {
        rest = after_table.trim_start();
    }
    if let Some(after_only) = strip_keyword_prefix_case_insensitive(rest, "ONLY") {
        rest = after_only.trim_start();
    }
    let mut restart_identity = false;
    if let Some(before_restart) = strip_keyword_suffix_case_insensitive(rest, "RESTART IDENTITY") {
        rest = before_restart.trim_end();
        restart_identity = true;
    }
    if rest.is_empty()
        || find_keyword_outside_quotes(rest, "CASCADE").is_some()
        || find_keyword_outside_quotes(rest, "RESTRICT").is_some()
        || find_keyword_outside_quotes(rest, "RESTART").is_some()
        || find_keyword_outside_quotes(rest, "CONTINUE").is_some()
        || find_keyword_outside_quotes(rest, "IDENTITY").is_some()
    {
        return Err(ParseError::InvalidRelationalSql);
    }
    let tables = split_csv(rest)?;
    let [table] = tables.as_slice() else {
        return Err(ParseError::InvalidRelationalSql);
    };
    Ok(TruncateTable {
        name: normalize_relation_identifier(table.trim())?,
        restart_identity,
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

fn split_optional_create_index_method(target: &str) -> Option<(&str, &str)> {
    let using_pos = find_keyword_outside_quotes(target, "USING")?;
    let table = target[..using_pos].trim();
    let method = target[using_pos + "USING".len()..].trim();
    (!table.is_empty() && !method.is_empty()).then_some((table, method))
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

fn parse_supported_sql_type_name(input: &str) -> Option<SqlType> {
    let ty = if input
        .get(.."pg_catalog.".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("pg_catalog."))
    {
        &input["pg_catalog.".len()..]
    } else {
        input
    };
    // Split off an optional `(...)` typmod (only NUMERIC/DECIMAL accept one).
    let (base, typmod) = match ty.find('(') {
        Some(open) => {
            let close = find_matching_paren(ty, open)?;
            if !ty[close + 1..].trim().is_empty() {
                return None;
            }
            (ty[..open].trim(), Some(ty[open + 1..close].trim()))
        }
        None => (ty.trim(), None),
    };
    if base.eq_ignore_ascii_case("INT")
        || base.eq_ignore_ascii_case("INT4")
        || base.eq_ignore_ascii_case("INTEGER")
    {
        typmod.is_none().then_some(SqlType::Int4)
    } else if base.eq_ignore_ascii_case("INT8") || base.eq_ignore_ascii_case("BIGINT") {
        typmod.is_none().then_some(SqlType::Int8)
    } else if base.eq_ignore_ascii_case("INT2") || base.eq_ignore_ascii_case("SMALLINT") {
        typmod.is_none().then_some(SqlType::Int2)
    } else if base.eq_ignore_ascii_case("NUMERIC") || base.eq_ignore_ascii_case("DECIMAL") {
        parse_numeric_typmod(typmod)
    } else if base.eq_ignore_ascii_case("BOOL") || base.eq_ignore_ascii_case("BOOLEAN") {
        typmod.is_none().then_some(SqlType::Bool)
    } else if base.eq_ignore_ascii_case("TEXT") {
        typmod.is_none().then_some(SqlType::Text)
    } else if base.eq_ignore_ascii_case("DATE") {
        typmod.is_none().then_some(SqlType::Date)
    } else if base.eq_ignore_ascii_case("TIMESTAMP") {
        // `timestamp` (without time zone); a fractional-second typmod is a follow-on.
        typmod.is_none().then_some(SqlType::Timestamp)
    } else if base.eq_ignore_ascii_case("UUID") {
        typmod.is_none().then_some(SqlType::Uuid)
    } else {
        None
    }
}

/// Parse a `numeric` typmod body (`"12,2"`, `"10"`, or absent) into a
/// `SqlType::Numeric { precision, scale }`. An absent typmod yields the
/// unconstrained default; precision must be 1..=38 (the i128 ceiling) and scale
/// 0..=precision, mirroring PostgreSQL's `numeric(p,s)` constraints.
fn parse_numeric_typmod(typmod: Option<&str>) -> Option<SqlType> {
    let Some(body) = typmod else {
        return Some(SqlType::Numeric {
            precision: NUMERIC_DEFAULT_PRECISION,
            scale: NUMERIC_DEFAULT_SCALE,
        });
    };
    let mut parts = body.split(',');
    let precision: u8 = parts.next()?.trim().parse().ok()?;
    let scale: u8 = match parts.next() {
        Some(scale) => scale.trim().parse().ok()?,
        None => 0,
    };
    if parts.next().is_some()
        || precision == 0
        || precision > NUMERIC_DEFAULT_PRECISION
        || scale > precision
    {
        return None;
    }
    Some(SqlType::Numeric { precision, scale })
}

fn parse_insert(input: &str) -> Result<Insert, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "INSERT")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "INTO"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let values_pos =
        find_keyword_outside_quotes(rest, "VALUES").ok_or(ParseError::InvalidRelationalSql)?;
    let target = rest[..values_pos].trim();
    let values = rest[values_pos + "VALUES".len()..].trim_start();
    let (table, columns) = if let Some(open) = target.find('(') {
        let close = find_matching_paren(target, open).ok_or(ParseError::InvalidRelationalSql)?;
        if !target[close + 1..].trim().is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        let columns = split_csv(&target[open + 1..close])?
            .into_iter()
            .map(|column| normalize_identifier(column.trim()))
            .collect::<Result<Vec<_>, _>>()?;
        if columns.is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        (
            normalize_relation_identifier(target[..open].trim())?,
            columns,
        )
    } else {
        (normalize_relation_identifier(target)?, Vec::new())
    };
    let mut rows = Vec::new();
    let mut tail = values;
    loop {
        let open = tail.find('(').ok_or(ParseError::InvalidRelationalSql)?;
        if !tail[..open].trim().is_empty() {
            return Err(ParseError::InvalidRelationalSql);
        }
        let close = find_matching_paren(tail, open).ok_or(ParseError::InvalidRelationalSql)?;
        let row = split_csv(&tail[open + 1..close])?
            .into_iter()
            .map(parse_sql_value)
            .collect::<Result<Vec<_>, _>>()?;
        if !columns.is_empty() && row.len() != columns.len() {
            return Err(ParseError::InvalidRelationalSql);
        }
        rows.push(row);
        tail = tail[close + 1..].trim_start();
        if tail.is_empty() {
            break;
        }
        let Some(after_comma) = tail.strip_prefix(',') else {
            return Err(ParseError::InvalidRelationalSql);
        };
        tail = after_comma.trim_start();
    }
    Ok(Insert {
        table,
        columns,
        rows,
    })
}

fn parse_delete(input: &str) -> Result<Delete, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "DELETE")
        .and_then(|s| strip_keyword_prefix_case_insensitive(s.trim_start(), "FROM"))
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let where_pos =
        find_keyword_outside_quotes(rest, "WHERE").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..where_pos].trim())?;
    let filter_input = rest[where_pos + "WHERE".len()..].trim();
    if filter_input.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let filter_groups = parse_select_filter_groups(filter_input)?;
    let filters = filter_groups.first().cloned().unwrap_or_default();
    Ok(Delete {
        table,
        filter: filters.first().cloned(),
        filters,
        filter_groups,
    })
}

fn parse_update(input: &str) -> Result<Update, ParseError> {
    let rest = strip_keyword_prefix_case_insensitive(input, "UPDATE")
        .ok_or(ParseError::InvalidRelationalSql)?
        .trim_start();
    let set_pos =
        find_keyword_outside_quotes(rest, "SET").ok_or(ParseError::InvalidRelationalSql)?;
    let table = normalize_relation_identifier(rest[..set_pos].trim())?;
    let after_set = rest[set_pos + "SET".len()..].trim_start();
    let where_pos =
        find_keyword_outside_quotes(after_set, "WHERE").ok_or(ParseError::InvalidRelationalSql)?;
    let assignment_input = after_set[..where_pos].trim();
    let filter_input = after_set[where_pos + "WHERE".len()..].trim();
    if assignment_input.is_empty() || filter_input.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let assignments = split_csv(assignment_input)?
        .into_iter()
        .map(parse_update_assignment)
        .collect::<Result<Vec<_>, _>>()?;
    if assignments.is_empty() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let filter_groups = parse_select_filter_groups(filter_input)?;
    let filters = filter_groups.first().cloned().unwrap_or_default();
    Ok(Update {
        table,
        assignments,
        filter: filters.first().cloned(),
        filters,
        filter_groups,
    })
}

fn parse_update_assignment(input: &str) -> Result<UpdateAssignment, ParseError> {
    let (column, value) = input
        .split_once('=')
        .ok_or(ParseError::InvalidRelationalSql)?;
    let column = normalize_identifier(column.trim())?;
    let value = parse_sql_value(value.trim())?;
    Ok(UpdateAssignment { column, value })
}

fn parse_sql_value(input: &str) -> Result<SqlValue, ParseError> {
    let (s, cast) = split_supported_sql_value_cast(input.trim())?;
    if s.starts_with('\'') {
        if !s.ends_with('\'') || s.len() < 2 {
            return Err(ParseError::InvalidRelationalSql);
        }
        let inner = &s[1..s.len() - 1];
        let value = inner.replace("''", "'");
        return match cast {
            None | Some(SqlType::Text) => Ok(SqlValue::Text(value)),
            Some(ty) => parse_typed_value_from_str(&value, ty),
        };
    }
    // Unquoted literal. A cast pins the target type; otherwise we infer it (a bare
    // integer stays Int4 as before — widening only kicks in for the new shapes:
    // `TRUE`/`FALSE` → Bool, a value with a decimal point → Numeric, and an integer
    // that overflows i32 → Int8).
    match cast {
        Some(ty) => parse_typed_value_from_str(s, ty),
        None => parse_inferred_unquoted_literal(s),
    }
}

/// Parse `text` into a specific [`SqlType`] (used for an explicit `::type` cast and
/// for a quoted literal carrying a cast). The numeric arm rounds to the column scale.
fn parse_typed_value_from_str(text: &str, ty: SqlType) -> Result<SqlValue, ParseError> {
    match ty {
        SqlType::Int2 => text
            .parse::<i16>()
            .map(SqlValue::Int2)
            .map_err(|_| ParseError::InvalidRelationalSql),
        SqlType::Int4 => text
            .parse::<i32>()
            .map(SqlValue::Int4)
            .map_err(|_| ParseError::InvalidRelationalSql),
        SqlType::Int8 => text
            .parse::<i64>()
            .map(SqlValue::Int8)
            .map_err(|_| ParseError::InvalidRelationalSql),
        SqlType::Numeric { scale, .. } => Decimal128::parse_at_scale(text, scale)
            .map(SqlValue::Numeric)
            .ok_or(ParseError::InvalidRelationalSql),
        SqlType::Bool => parse_bool_value(text)
            .map(SqlValue::Bool)
            .ok_or(ParseError::InvalidRelationalSql),
        SqlType::Text => Ok(SqlValue::Text(text.to_string())),
        SqlType::Date => crate::datetime::parse_date(text)
            .map(SqlValue::Date)
            .ok_or(ParseError::InvalidRelationalSql),
        SqlType::Timestamp => crate::datetime::parse_timestamp(text)
            .map(SqlValue::Timestamp)
            .ok_or(ParseError::InvalidRelationalSql),
        SqlType::Uuid => crate::uuid::parse_uuid(text)
            .map(SqlValue::Uuid)
            .ok_or(ParseError::InvalidRelationalSql),
    }
}

/// Infer a [`SqlValue`] from an unquoted, uncast literal. Preserves the pre-existing
/// rule that a bare integer is `Int4`; widens only to the genuinely new shapes.
fn parse_inferred_unquoted_literal(s: &str) -> Result<SqlValue, ParseError> {
    // The unquoted `NULL` keyword is the typeless SQL null (a quoted `'NULL'` is the text value,
    // handled on the quoted path). Valid for any column type; coercion passes it through unchanged.
    if s.eq_ignore_ascii_case("NULL") {
        return Ok(SqlValue::Null);
    }
    if s.eq_ignore_ascii_case("TRUE") {
        return Ok(SqlValue::Bool(true));
    }
    if s.eq_ignore_ascii_case("FALSE") {
        return Ok(SqlValue::Bool(false));
    }
    if let Ok(value) = s.parse::<i32>() {
        return Ok(SqlValue::Int4(value));
    }
    // A decimal point means NUMERIC; carry the literal's natural scale (its fractional
    // digit count) so `1.00` keeps scale 2 until the engine rescales to the column.
    if s.contains('.') {
        if let Some(decimal) = Decimal128::parse(s) {
            return Ok(SqlValue::Numeric(decimal));
        }
        return Err(ParseError::InvalidRelationalSql);
    }
    // An integer too wide for i32 widens to Int8 (rather than the old hard error).
    s.parse::<i64>()
        .map(SqlValue::Int8)
        .map_err(|_| ParseError::InvalidRelationalSql)
}

fn split_supported_sql_value_cast(input: &str) -> Result<(&str, Option<SqlType>), ParseError> {
    let Some(pos) = find_cast_operator_outside_quotes(input) else {
        return Ok((input, None));
    };
    let value = input[..pos].trim();
    let ty = input[pos + 2..].trim();
    if value.is_empty() || ty.is_empty() || find_cast_operator_outside_quotes(ty).is_some() {
        return Err(ParseError::InvalidRelationalSql);
    }
    let ty = if ty
        .get(.."pg_catalog.".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("pg_catalog."))
    {
        &ty["pg_catalog.".len()..]
    } else {
        ty
    };
    let cast = parse_supported_sql_type_name(ty).ok_or(ParseError::InvalidRelationalSql)?;
    Ok((value, Some(cast)))
}

fn find_cast_operator_outside_quotes(input: &str) -> Option<usize> {
    let bytes = input.as_bytes();
    let mut in_quote = false;
    let mut idx = 0;
    while idx + 1 < bytes.len() {
        match bytes[idx] {
            b'\'' => {
                if in_quote && bytes.get(idx + 1) == Some(&b'\'') {
                    idx += 1;
                } else {
                    in_quote = !in_quote;
                }
            }
            b':' if !in_quote && bytes[idx + 1] == b':' => return Some(idx),
            _ => {}
        }
        idx += 1;
    }
    None
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

fn parse_set_session_command(rest: &str) -> Option<Result<Command, ParseError>> {
    if let Some(after_local) = strip_keyword_prefix_case_insensitive(rest, "LOCAL") {
        let after_local = after_local.trim_start();
        if let Some(after_role) = strip_keyword_prefix_case_insensitive(after_local, "ROLE") {
            let tail = after_role.trim_start();
            return Some(parse_set_role_command(tail));
        }

        if let Some(after_transaction) =
            strip_keyword_prefix_case_insensitive(after_local, "TRANSACTION")
        {
            let tail = after_transaction.trim_start();
            return Some(
                if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }
    }

    if let Some(after_session) = strip_keyword_prefix_case_insensitive(rest, "SESSION") {
        let after_session = after_session.trim_start();
        if let Some(after_role) = strip_keyword_prefix_case_insensitive(after_session, "ROLE") {
            let tail = after_role.trim_start();
            return Some(parse_set_role_command(tail));
        }
    }

    if let Some(after_role) = strip_keyword_prefix_case_insensitive(rest, "ROLE") {
        let tail = after_role.trim_start();
        return Some(parse_set_role_command(tail));
    }

    if let Some(after_transaction) = strip_keyword_prefix_case_insensitive(rest, "TRANSACTION") {
        let tail = after_transaction.trim_start();
        return Some(
            if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                Ok(Command::ResetAll)
            } else {
                Err(ParseError::InvalidSet)
            },
        );
    }

    if let Some(after_session) = strip_keyword_prefix_case_insensitive(rest, "SESSION") {
        let after_session = after_session.trim_start();

        if let Some(after_authorization) =
            strip_keyword_prefix_case_insensitive(after_session, "AUTHORIZATION")
        {
            let tail = after_authorization.trim_start();
            return Some(
                if parse_reset_identifier(tail)
                    .is_some_and(|(_, trailing)| trailing.trim().is_empty())
                {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }

        if let Some(after_auth) = strip_keyword_prefix_case_insensitive(after_session, "AUTH") {
            let tail = after_auth.trim_start();
            return Some(
                if parse_reset_identifier(tail)
                    .is_some_and(|(_, trailing)| trailing.trim().is_empty())
                {
                    Ok(Command::ResetAll)
                } else {
                    Err(ParseError::InvalidSet)
                },
            );
        }

        if let Some(after_characteristics) =
            strip_keyword_prefix_case_insensitive(after_session, "CHARACTERISTICS")
        {
            let after_characteristics = after_characteristics.trim_start();
            if let Some(after_as) =
                strip_keyword_prefix_case_insensitive(after_characteristics, "AS")
            {
                let after_as = after_as.trim_start();
                if let Some(after_transaction) =
                    strip_keyword_prefix_case_insensitive(after_as, "TRANSACTION")
                {
                    let tail = after_transaction.trim_start();
                    return Some(
                        if !tail.is_empty() && is_begin_mode_list(&normalize_begin_tokens(tail)) {
                            Ok(Command::ResetAll)
                        } else {
                            Err(ParseError::InvalidSet)
                        },
                    );
                }
            }
            return Some(Err(ParseError::InvalidSet));
        }

        return None;
    }

    None
}

fn parse_set_role_command(tail: &str) -> Result<Command, ParseError> {
    if tail.eq_ignore_ascii_case("NONE") || tail.eq_ignore_ascii_case("DEFAULT") {
        return Ok(Command::SetRole { role: None });
    }
    if let Some((role, trailing)) = parse_reset_identifier(tail) {
        if trailing.trim().is_empty() {
            return Ok(Command::SetRole {
                role: Some(normalize_identifier(role)?),
            });
        }
    }
    Err(ParseError::InvalidSet)
}

fn is_isolation_level_suffix(tokens: &[&str]) -> bool {
    matches!(
        tokens,
        [isolation, level, serializable]
            if isolation.eq_ignore_ascii_case("ISOLATION")
                && level.eq_ignore_ascii_case("LEVEL")
                && serializable.eq_ignore_ascii_case("SERIALIZABLE")
    ) || matches!(
        tokens,
        [isolation, level, repeatable, read]
            if isolation.eq_ignore_ascii_case("ISOLATION")
                && level.eq_ignore_ascii_case("LEVEL")
                && repeatable.eq_ignore_ascii_case("REPEATABLE")
                && read.eq_ignore_ascii_case("READ")
    ) || matches!(
        tokens,
        [isolation, level, read, committed]
            if isolation.eq_ignore_ascii_case("ISOLATION")
                && level.eq_ignore_ascii_case("LEVEL")
                && read.eq_ignore_ascii_case("READ")
                && committed.eq_ignore_ascii_case("COMMITTED")
    ) || matches!(
        tokens,
        [isolation, level, read, uncommitted]
            if isolation.eq_ignore_ascii_case("ISOLATION")
                && level.eq_ignore_ascii_case("LEVEL")
                && read.eq_ignore_ascii_case("READ")
                && uncommitted.eq_ignore_ascii_case("UNCOMMITTED")
    )
}

fn is_deferrable_suffix(tokens: &[&str]) -> bool {
    matches!(
        tokens,
        [deferrable] if deferrable.eq_ignore_ascii_case("DEFERRABLE")
    ) || matches!(
        tokens,
        [not, deferrable]
            if not.eq_ignore_ascii_case("NOT") && deferrable.eq_ignore_ascii_case("DEFERRABLE")
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BeginModeKind {
    AccessMode,
    IsolationLevel,
    Deferrable,
}

fn parse_begin_mode(tokens: &[String]) -> Option<(usize, BeginModeKind)> {
    let refs: Vec<_> = tokens.iter().map(String::as_str).collect();

    if refs.len() >= 4 && is_isolation_level_suffix(&refs[..4]) {
        return Some((4, BeginModeKind::IsolationLevel));
    }

    if refs.len() >= 3 && is_isolation_level_suffix(&refs[..3]) {
        return Some((3, BeginModeKind::IsolationLevel));
    }

    if refs.len() >= 2 {
        let two = &refs[..2];
        if matches!(
            two,
            [read, only]
                if read.eq_ignore_ascii_case("READ") && only.eq_ignore_ascii_case("ONLY")
        ) || matches!(
            two,
            [read, write]
                if read.eq_ignore_ascii_case("READ") && write.eq_ignore_ascii_case("WRITE")
        ) {
            return Some((2, BeginModeKind::AccessMode));
        }

        if matches!(
            two,
            [not, deferrable]
                if not.eq_ignore_ascii_case("NOT") && deferrable.eq_ignore_ascii_case("DEFERRABLE")
        ) {
            return Some((2, BeginModeKind::Deferrable));
        }
    }

    if !refs.is_empty() && is_deferrable_suffix(&refs[..1]) {
        return Some((1, BeginModeKind::Deferrable));
    }

    None
}

fn normalize_begin_tokens(input: &str) -> Vec<String> {
    let mut normalized = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        if ch == ',' {
            normalized.push(' ');
            normalized.push(',');
            normalized.push(' ');
        } else {
            normalized.push(ch);
        }
    }
    normalized.split_whitespace().map(str::to_owned).collect()
}

fn is_begin_mode_list(tokens: &[String]) -> bool {
    if tokens.is_empty() {
        return false;
    }

    let mut idx = 0;
    let mut seen_access_mode = false;
    let mut seen_isolation_level = false;
    let mut seen_deferrable = false;

    while idx < tokens.len() {
        if tokens[idx] == "," {
            return false;
        }

        let Some((consumed, kind)) = parse_begin_mode(&tokens[idx..]) else {
            return false;
        };

        match kind {
            BeginModeKind::AccessMode if seen_access_mode => return false,
            BeginModeKind::IsolationLevel if seen_isolation_level => return false,
            BeginModeKind::Deferrable if seen_deferrable => return false,
            BeginModeKind::AccessMode => seen_access_mode = true,
            BeginModeKind::IsolationLevel => seen_isolation_level = true,
            BeginModeKind::Deferrable => seen_deferrable = true,
        }

        idx += consumed;
        if idx == tokens.len() {
            return true;
        }

        if tokens[idx] == "," {
            idx += 1;
            if idx == tokens.len() || tokens[idx] == "," {
                return false;
            }
        }
    }

    true
}

fn strip_single_leading_comma(tokens: &[String]) -> Option<&[String]> {
    match tokens {
        [first, rest @ ..] if first == "," && !rest.is_empty() => Some(rest),
        _ => None,
    }
}

fn is_begin_with_optional_mode(input: &str) -> bool {
    let tokens = normalize_begin_tokens(input);
    let Some((first, rest)) = tokens.split_first() else {
        return false;
    };

    if first.eq_ignore_ascii_case("BEGIN") {
        return match rest {
            [] => true,
            [second] if second.eq_ignore_ascii_case("TRANSACTION") => true,
            [second] if second.eq_ignore_ascii_case("WORK") => true,
            mode if is_begin_mode_list(mode) => true,
            [second, mode @ ..]
                if second.eq_ignore_ascii_case("TRANSACTION")
                    || second.eq_ignore_ascii_case("WORK") =>
            {
                if is_begin_mode_list(mode) {
                    true
                } else {
                    strip_single_leading_comma(mode).is_some_and(is_begin_mode_list)
                }
            }
            _ => false,
        };
    }

    if first.eq_ignore_ascii_case("START") {
        return match rest {
            [second] if second.eq_ignore_ascii_case("TRANSACTION") => true,
            [second] if second.eq_ignore_ascii_case("WORK") => true,
            [second, mode @ ..]
                if second.eq_ignore_ascii_case("TRANSACTION")
                    || second.eq_ignore_ascii_case("WORK") =>
            {
                if is_begin_mode_list(mode) {
                    true
                } else {
                    strip_single_leading_comma(mode).is_some_and(is_begin_mode_list)
                }
            }
            _ => false,
        };
    }

    false
}

/// Parse a single SQL command (the strict, public entry). A SELECT whose FROM target is a
/// `pg_catalog.`/`information_schema.`-qualified relation is rejected, exactly as before —
/// the legacy compatibility server relies on this (catalog queries fail to parse and route
/// to its compatibility layer). The engine uses [`parse_command_allowing_catalog`] instead.
pub fn parse_command(input: &str) -> Result<Command, ParseError> {
    parse_command_inner(input, false)
}

/// Like [`parse_command`] but carries a `pg_catalog.`/`information_schema.` qualifier on a
/// SELECT's FROM target through to the parsed `Select`, for the engine's native catalog
/// support (Phase-3 M2). Behaves identically to [`parse_command`] for every other command.
pub fn parse_command_allowing_catalog(input: &str) -> Result<Command, ParseError> {
    parse_command_inner(input, true)
}

fn parse_command_inner(input: &str, allow_catalog_schemas: bool) -> Result<Command, ParseError> {
    let mut s = input.trim_end();
    if s.is_empty() {
        return Err(ParseError::Empty);
    }

    while let Some(without_semicolon) = s.strip_suffix(';') {
        s = without_semicolon.trim_end();
    }

    let s = s.trim_start();
    if s.is_empty() {
        return Err(ParseError::Empty);
    }

    if is_begin_with_optional_mode(s) {
        return Ok(Command::Begin);
    }
    if let Some(chain) = parse_transaction_control_chain(s, "COMMIT")
        .or_else(|| parse_transaction_control_chain(s, "END"))
    {
        return Ok(Command::Commit { chain });
    }
    if let Some(chain) = parse_transaction_control_chain(s, "ROLLBACK")
        .or_else(|| parse_transaction_control_chain(s, "ABORT"))
    {
        return Ok(Command::Rollback { chain });
    }
    if let Some(flush) = parse_flush_command(s) {
        return Ok(flush);
    }
    if let Some(reset) = parse_reset_command(s) {
        return reset;
    }
    if let Some(relational) = parse_relational_command(s, allow_catalog_schemas) {
        return relational;
    }

    let mut parts = s.splitn(2, char::is_whitespace);
    if let Some(cmd) = parts.next() {
        if cmd.eq_ignore_ascii_case("SET") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidSet);
            };
            if let Some(alias) = parse_set_session_command(rest) {
                return alias;
            }

            let assignment_rest = strip_set_scope_prefix(rest, "LOCAL")
                .or_else(|| strip_set_scope_prefix(rest, "SESSION"))
                .unwrap_or(rest);
            let Some((k, v)) = split_set_key_value(assignment_rest) else {
                return Err(ParseError::InvalidSet);
            };
            let key = k.trim();
            let value = v.trim();
            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(ParseError::InvalidSet);
            }
            return Ok(Command::SetKv {
                key: key.to_string(),
                value: value.to_string(),
            });
        }

        if cmd.eq_ignore_ascii_case("DEL") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidDel);
            };
            let key = rest.trim();
            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(ParseError::InvalidDel);
            }
            return Ok(Command::DeleteKv {
                key: key.to_string(),
            });
        }

        if cmd.eq_ignore_ascii_case("DELETE") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidDel);
            };
            let rest = rest.trim();
            let key = if let Some((prefix, remainder)) = rest.split_once(char::is_whitespace) {
                if prefix.eq_ignore_ascii_case("FROM") {
                    let candidate = remainder.trim();
                    if candidate.is_empty() || candidate.chars().any(char::is_whitespace) {
                        return Err(ParseError::InvalidDel);
                    }
                    candidate
                } else {
                    return Err(ParseError::InvalidDel);
                }
            } else if rest.eq_ignore_ascii_case("FROM") {
                return Err(ParseError::InvalidDel);
            } else {
                rest
            };

            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(ParseError::InvalidDel);
            }

            return Ok(Command::DeleteKv {
                key: key.to_string(),
            });
        }

        if cmd.eq_ignore_ascii_case("GET") {
            let Some(rest) = parts.next() else {
                return Err(ParseError::InvalidGet);
            };
            let key = rest.trim();
            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(ParseError::InvalidGet);
            }
            return Ok(Command::GetKv {
                key: key.to_string(),
            });
        }
    }

    Err(ParseError::Unsupported(s.to_string()))
}

#[cfg(test)]
mod decimal_tests {
    use super::*;

    #[test]
    fn parses_supported_type_names_including_typmod() {
        assert_eq!(parse_supported_sql_type_name("INT"), Some(SqlType::Int4));
        assert_eq!(
            parse_supported_sql_type_name("integer"),
            Some(SqlType::Int4)
        );
        assert_eq!(parse_supported_sql_type_name("BIGINT"), Some(SqlType::Int8));
        assert_eq!(parse_supported_sql_type_name("int8"), Some(SqlType::Int8));
        assert_eq!(parse_supported_sql_type_name("text"), Some(SqlType::Text));
        assert_eq!(parse_supported_sql_type_name("BOOL"), Some(SqlType::Bool));
        assert_eq!(
            parse_supported_sql_type_name("boolean"),
            Some(SqlType::Bool)
        );
        assert_eq!(
            parse_supported_sql_type_name("NUMERIC(12,2)"),
            Some(SqlType::Numeric {
                precision: 12,
                scale: 2
            })
        );
        assert_eq!(
            parse_supported_sql_type_name("numeric(12, 2)"),
            Some(SqlType::Numeric {
                precision: 12,
                scale: 2
            })
        );
        assert_eq!(
            parse_supported_sql_type_name("DECIMAL(10)"),
            Some(SqlType::Numeric {
                precision: 10,
                scale: 0
            })
        );
        assert_eq!(
            parse_supported_sql_type_name("NUMERIC"),
            Some(SqlType::Numeric {
                precision: NUMERIC_DEFAULT_PRECISION,
                scale: NUMERIC_DEFAULT_SCALE
            })
        );
        // A typmod on a non-numeric type, or an out-of-range numeric typmod, is rejected.
        assert_eq!(parse_supported_sql_type_name("INT(4)"), None);
        assert_eq!(parse_supported_sql_type_name("NUMERIC(0,0)"), None);
        assert_eq!(parse_supported_sql_type_name("NUMERIC(2,5)"), None);
        assert_eq!(parse_supported_sql_type_name("NUMERIC(99)"), None);
    }

    #[test]
    fn parses_typed_literals_and_casts() {
        // Bare literal inference: integer stays Int4, decimal becomes Numeric, TRUE/FALSE bool.
        assert_eq!(parse_sql_value("5").unwrap(), SqlValue::Int4(5));
        assert_eq!(
            parse_sql_value("1.50").unwrap(),
            SqlValue::Numeric(Decimal128::new(150, 2))
        );
        assert_eq!(parse_sql_value("TRUE").unwrap(), SqlValue::Bool(true));
        assert_eq!(parse_sql_value("false").unwrap(), SqlValue::Bool(false));
        // An integer beyond i32 widens to Int8.
        assert_eq!(
            parse_sql_value("9000000000").unwrap(),
            SqlValue::Int8(9_000_000_000)
        );
        // Casts pin the type (numeric cast rounds to the typmod scale).
        assert_eq!(
            parse_sql_value("'1.005'::numeric(12,2)").unwrap(),
            SqlValue::Numeric(Decimal128::new(101, 2))
        );
        assert_eq!(parse_sql_value("'t'::bool").unwrap(), SqlValue::Bool(true));
        assert_eq!(parse_sql_value("'42'::int8").unwrap(), SqlValue::Int8(42));
    }
}
