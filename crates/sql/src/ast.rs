//! SQL command and relational DDL/DML abstract syntax contracts.

use super::{
    DatabasePrivileges, DefaultTablePrivileges, FunctionPrivileges, GrantTable, RevokeTable,
    SchemaPrivileges, Select, SelectFilter, SqlType, SqlValue, TablespacePrivileges,
};

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
