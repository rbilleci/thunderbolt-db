//! SQL command and relational DDL/DML abstract syntax contracts.

use super::{
    DatabasePrivileges, DefaultTablePrivileges, FunctionPrivileges, GrantTable, RevokeTable,
    SchemaPrivileges, Select, SelectFilter, SqlType, SqlValue, TablespacePrivileges,
};

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionIsolation {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionAccessMode {
    ReadWrite,
    ReadOnly,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransactionCharacteristics {
    pub isolation: TransactionIsolation,
    pub access: TransactionAccessMode,
    pub deferrable: bool,
}

impl Default for TransactionCharacteristics {
    fn default() -> Self {
        Self {
            isolation: TransactionIsolation::ReadCommitted,
            access: TransactionAccessMode::ReadWrite,
            deferrable: false,
        }
    }
}

impl TransactionCharacteristics {
    pub const READ_COMMITTED_READ_WRITE: Self = Self {
        isolation: TransactionIsolation::ReadCommitted,
        access: TransactionAccessMode::ReadWrite,
        deferrable: false,
    };

    pub const REPEATABLE_READ_WRITE: Self = Self {
        isolation: TransactionIsolation::RepeatableRead,
        access: TransactionAccessMode::ReadWrite,
        deferrable: false,
    };
}

/// Typed SQL command ownership.
///
/// New variants stay append-only. Canonical WAL uses named JSON tags, but changing established
/// in-memory discriminants also changes large engine match tables and hot linked-code layout; a
/// mid-enum insertion measurably regressed the production point-read route.
#[allow(clippy::large_enum_variant)]
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Begin {
        characteristics: TransactionCharacteristics,
    },
    Commit {
        chain: bool,
    },
    Rollback {
        chain: bool,
    },
    Flush,
    ResetAll,
    SetRole {
        role: Option<String>,
    },
    SetKv {
        key: String,
        value: String,
    },
    DeleteKv {
        key: String,
    },
    GetKv {
        key: String,
    },
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
    SelectLiteral(SelectLiteral),
    ShowTransactionIsolation,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateSchema {
    pub name: String,
    pub if_not_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropSchema {
    pub name: String,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateDatabase {
    pub name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropDatabase {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameDatabase {
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateTablespace {
    pub name: String,
    pub location: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropTablespace {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameTablespace {
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateTable {
    pub table: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Option<PrimaryKey>,
    pub unique_constraints: Vec<UniqueConstraint>,
    pub check_constraints: Vec<CheckConstraint>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AddPrimaryKey {
    pub table: String,
    pub name: String,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND key has len > 1.
    pub columns: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PrimaryKey {
    pub name: Option<String>,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND primary key has len > 1.
    pub columns: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct UniqueConstraint {
    pub name: Option<String>,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND unique constraint has len > 1.
    pub columns: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AddUniqueConstraint {
    pub table: String,
    pub name: String,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND unique constraint has len > 1.
    pub columns: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CheckConstraint {
    pub name: Option<String>,
    pub filter: SelectFilter,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AddCheckConstraint {
    pub table: String,
    pub name: String,
    pub filter: SelectFilter,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AddForeignKey {
    pub table: String,
    pub name: String,
    pub column: String,
    pub referenced_table: String,
    pub referenced_column: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AddColumn {
    pub table: String,
    pub column: ColumnDef,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropColumn {
    pub table: String,
    pub column: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameTable {
    pub old_name: String,
    pub new_name: String,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameColumn {
    pub table: String,
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameConstraint {
    pub table: String,
    pub old_name: String,
    pub new_name: String,
    pub table_if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropConstraint {
    pub table: String,
    pub name: String,
    pub table_if_exists: bool,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateIndex {
    pub name: String,
    pub table: String,
    /// The FIRST key column (== `columns[0]`); kept for single-column call sites.
    pub column: String,
    /// The ordered key columns (>= 1). A COMPOUND index/constraint has len > 1.
    pub columns: Vec<String>,
    pub unique: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameIndex {
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateView {
    pub name: String,
    pub query: Select,
    pub definition: String,
    pub or_replace: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameView {
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateMaterializedView {
    pub name: String,
    pub query: Select,
    pub definition: String,
    pub with_data: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RefreshMaterializedView {
    pub name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameMaterializedView {
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateFunction {
    pub name: String,
    pub return_type: SqlType,
    pub body: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameFunction {
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropFunction {
    pub name: String,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SelectFunction {
    pub name: String,
}

/// One bounded, no-`FROM` scalar projection such as `SELECT 1 AS one`.
///
/// The parser resolves the literal's PostgreSQL type up front, so execution can materialize one
/// typed transient relation on the GPU without carrying SQL text or wire metadata into the engine.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SelectLiteral {
    pub column_name: String,
    pub ty: SqlType,
    pub value: SqlValue,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateSequence {
    pub name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SequenceNextVal {
    pub name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SequenceCurrVal {
    pub name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SequenceSetVal {
    pub name: String,
    pub value: i64,
    pub is_called: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameSequence {
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropSequence {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum PublicationTarget {
    AllTables,
    Tables(Vec<String>),
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreatePublication {
    pub name: String,
    pub target: PublicationTarget,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropPublication {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateSubscription {
    pub name: String,
    pub connection: String,
    pub publications: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropSubscription {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateRole {
    pub name: String,
    pub login: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropRole {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RenameRole {
    pub old_name: String,
    pub new_name: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropTable {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TruncateTable {
    pub name: String,
    pub restart_identity: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropIndex {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropMaterializedView {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropView {
    pub names: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AlterColumnDefault {
    pub table: String,
    pub column: String,
    pub default: Option<ColumnDefault>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CommentOn {
    pub target: CommentTarget,
    pub comment: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
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

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub ty: SqlType,
    pub domain: Option<String>,
    pub default: Option<ColumnDefault>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateDomain {
    pub name: String,
    pub base_type: SqlType,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropDomain {
    pub domains: Vec<String>,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CreateExtension {
    pub name: String,
    pub if_not_exists: bool,
    pub schema: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DropExtension {
    pub name: String,
    pub if_exists: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum ColumnDefault {
    Literal(SqlValue),
    SequenceNextVal {
        sequence: String,
        create_if_missing: bool,
    },
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Insert {
    pub table: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<SqlValue>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub returning: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Delete {
    pub table: String,
    pub filter: Option<SelectFilter>,
    pub filters: Vec<SelectFilter>,
    pub filter_groups: Vec<Vec<SelectFilter>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub returning: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Update {
    pub table: String,
    pub assignments: Vec<UpdateAssignment>,
    pub filter: Option<SelectFilter>,
    pub filters: Vec<SelectFilter>,
    pub filter_groups: Vec<Vec<SelectFilter>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub returning: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct UpdateAssignment {
    pub column: String,
    /// `Some(column)` lowers the bounded checked form `target = column + value`.
    /// PRODUCT-002 accepts only `target == column`; all other expression shapes fail parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_column: Option<String>,
    pub value: SqlValue,
}
