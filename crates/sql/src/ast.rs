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
        #[serde(default)]
        scope: SetRoleScope,
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
    /// Session control accepted by the product protocol boundary. `transaction` carries the exact
    /// characteristics for `SET TRANSACTION`; `access_share_relations` carries the bounded
    /// relation list for pg_dump's `LOCK TABLE ... IN ACCESS SHARE MODE`. Empty fields describe a
    /// bounded PostgreSQL client setting whose state does not affect engine name/type semantics.
    ///
    /// Keep this variant append-only: command discriminant order is performance-sensitive.
    SessionControl {
        transaction: Option<TransactionCharacteristics>,
        access_share_relations: Vec<String>,
    },
    /// Strict, versioned prepared catalog programs emitted by supported PostgreSQL clients. The
    /// parameter remains a typed AST value and is bound without reconstructing or reparsing SQL.
    PreparedCatalog(PreparedCatalogProgram),
    /// Bounded role-login mutation emitted by PostgreSQL 16 pg_dumpall. Keep append-only: command
    /// discriminant order is performance-sensitive.
    AlterRoleLogin(AlterRoleLogin),
    /// Transactional sequence value-overlay reset. Keep append-only: command discriminant order
    /// is performance-sensitive, and ordinary `setval` has deliberately different rollback
    /// semantics.
    SequenceRestart(SequenceRestart),
}

/// PostgreSQL role-setting lifetime.  `SESSION` is also the bare/default `SET ROLE` scope;
/// `LOCAL` is transaction-local and therefore must remain explicit in the executable AST.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SetRoleScope {
    #[default]
    Session,
    Local,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum PreparedCatalogProgram {
    Pg16DomainConstraints { type_oid: SqlValue },
    Pg16DomainDefinition { type_oid: SqlValue },
    Pg16FunctionDefinition { function_oid: SqlValue },
    Pg16MaterializedViewDependencies,
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
    /// CHECK-only input typing. Older AST/WAL payloads omit this field and deliberately retain
    /// `LegacyAmbiguous` rather than being guessed as a new SQL spelling.
    #[serde(
        default,
        skip_serializing_if = "CheckLiteralProvenance::is_legacy_ambiguous"
    )]
    pub literal_provenance: CheckLiteralProvenance,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AddCheckConstraint {
    pub table: String,
    pub name: String,
    pub filter: SelectFilter,
    /// See [`CheckConstraint::literal_provenance`].
    #[serde(
        default,
        skip_serializing_if = "CheckLiteralProvenance::is_legacy_ambiguous"
    )]
    pub literal_provenance: CheckLiteralProvenance,
}

/// The SQL input type of a CHECK predicate literal. This intentionally lives alongside CHECK
/// nodes instead of widening generic SELECT filters: uncast quoted literals and `NULL` are SQL
/// `unknown` only at the comparison site, while a `SqlValue::Text` alone cannot distinguish that
/// spelling from explicit `::text`.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckLiteralProvenance {
    /// Historical AST/WAL omitted provenance. Replay applies a narrowly documented compatibility
    /// rule, never treating it as evidence that the original spelling was uncast/unknown.
    #[default]
    LegacyAmbiguous,
    /// An uncast quoted literal or uncast NULL, bound at the CHECK comparison target.
    Unknown,
    /// An inferred non-text scalar or explicit `::type` cast.
    Known(SqlType),
}

impl CheckLiteralProvenance {
    pub const fn is_legacy_ambiguous(&self) -> bool {
        matches!(self, Self::LegacyAmbiguous)
    }
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
/// A prepared `int4` parameter plus a bounded `int4` constant is folded at Bind, once all
/// parameters are typed, into that same transient GPU scalar relation.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SelectLiteral {
    pub column_name: String,
    pub ty: SqlType,
    pub value: SqlValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_int4: Option<i32>,
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
pub struct SequenceRestart {
    pub name: String,
    pub value: i64,
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

/// Bounded role-attribute mutation used by PostgreSQL 16 global dumps. The engine's role model
/// deliberately owns login capability only; the parser accepts pg_dump's complete fixed-default
/// attribute program but rejects any non-default privilege attribute before mutation admission.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AlterRoleLogin {
    pub name: String,
    pub login: bool,
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
    /// A scalar SQL DEFAULT whose input type is retained until assignment evaluation.
    ///
    /// SQL parsing writes this variant instead of eagerly collapsing a value to the
    /// column type.  That preserves source `numeric(p,s)` checks and makes CREATE /
    /// ALTER publication independent from a later INSERT failure.  `Literal` remains
    /// the historical/programmatic compatibility representation and its serde shape
    /// must not change.
    DeferredScalar {
        value: SqlValue,
        input: DefaultInputType,
    },
    Literal(SqlValue),
    SequenceNextVal {
        sequence: String,
        create_if_missing: bool,
    },
}

/// Type authority carried by a deferred scalar column DEFAULT.
///
/// `Unknown` is an in-flight parser representation for `ALTER ... SET DEFAULT`;
/// binding to the target column normalizes it to `TargetTyped` before catalog
/// publication.  The other variants are durable because they affect source-side
/// assignment checks (notably `numeric(p,s)`).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultInputType {
    Unknown,
    TargetTyped,
    Inferred(SqlType),
    Explicit(SqlType),
}

/// Source authority for a supplied INSERT value.
///
/// Live parsed/bound commands retain this provenance through semantic preparation. Historical
/// typed-command JSON predates it: decoding an old raw `SqlValue` row deliberately labels the
/// value `Programmatic`, because recovery must not invent a literal/bind distinction that was
/// never persisted. The canonical typed INSERT envelope will carry resolved semantics directly;
/// it must not rely on reserializing this AST to recover provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertValueProvenance {
    Literal,
    Parameter { index: usize },
    BoundParameter { index: usize },
    Programmatic,
}

/// Source authority for an explicit INSERT `DEFAULT` request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertDefaultProvenance {
    SqlKeyword,
    Programmatic,
}

/// One syntactic INSERT cell.
///
/// `DEFAULT` is an expression request, not a scalar value and therefore must not be represented
/// by a `SqlValue` sentinel. The value/provenance pair remains one authority from parse through
/// Bind; an explicit parameter becomes `BoundParameter` without rendering or reparsing SQL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertCell {
    Value {
        value: SqlValue,
        provenance: InsertValueProvenance,
    },
    Default {
        provenance: InsertDefaultProvenance,
    },
}

impl InsertCell {
    pub fn literal(value: SqlValue) -> Self {
        Self::Value {
            value,
            provenance: InsertValueProvenance::Literal,
        }
    }

    pub fn programmatic(value: SqlValue) -> Self {
        Self::Value {
            value,
            provenance: InsertValueProvenance::Programmatic,
        }
    }

    pub fn sql_default() -> Self {
        Self::Default {
            provenance: InsertDefaultProvenance::SqlKeyword,
        }
    }

    pub fn programmatic_default() -> Self {
        Self::Default {
            provenance: InsertDefaultProvenance::Programmatic,
        }
    }

    pub(crate) fn from_parsed_value(value: SqlValue) -> Self {
        match value {
            SqlValue::Parameter { index, cast } => Self::Value {
                value: SqlValue::Parameter { index, cast },
                provenance: InsertValueProvenance::Parameter { index },
            },
            value => Self::literal(value),
        }
    }

    pub(crate) fn parameter_slot(&self) -> Option<(usize, Option<SqlType>)> {
        let Self::Value {
            value: SqlValue::Parameter { index, cast },
            provenance:
                InsertValueProvenance::Parameter {
                    index: provenance_index,
                },
        } = self
        else {
            return None;
        };
        (*index == *provenance_index).then_some((*index, *cast))
    }

    pub(crate) fn bind_parameter(&mut self, value: SqlValue, index: usize) {
        *self = Self::Value {
            value,
            provenance: InsertValueProvenance::BoundParameter { index },
        };
    }

    pub fn value(&self) -> Option<&SqlValue> {
        match self {
            Self::Value { value, .. } => Some(value),
            Self::Default { .. } => None,
        }
    }

    pub fn value_provenance(&self) -> Option<InsertValueProvenance> {
        match self {
            Self::Value { provenance, .. } => Some(*provenance),
            Self::Default { .. } => None,
        }
    }

    pub fn default_provenance(&self) -> Option<InsertDefaultProvenance> {
        match self {
            Self::Value { .. } => None,
            Self::Default { provenance } => Some(*provenance),
        }
    }
}

impl Insert {
    /// Test/parser expectation helper for literal scalar cells. SQL parsing itself also records
    /// literals with this provenance, while parsed parameters use their dedicated provenance.
    pub fn literal_rows(rows: Vec<Vec<SqlValue>>) -> Vec<Vec<InsertCell>> {
        rows.into_iter()
            .map(|row| row.into_iter().map(InsertCell::literal).collect())
            .collect()
    }

    /// Compatibility/programmatic ingress helper. It creates the same one-cell representation
    /// as the parser, but truthfully marks values that did not originate from SQL text or Bind.
    pub fn programmatic_rows(rows: Vec<Vec<SqlValue>>) -> Vec<Vec<InsertCell>> {
        rows.into_iter()
            .map(|row| row.into_iter().map(InsertCell::programmatic).collect())
            .collect()
    }
}

impl From<SqlValue> for InsertCell {
    fn from(value: SqlValue) -> Self {
        Self::programmatic(value)
    }
}

/// A reserved object shape that cannot collide with serde's externally tagged `SqlValue` enum.
/// The dollar-prefixed key is intentionally reserved for this AST transport only.
#[derive(serde::Serialize, serde::Deserialize)]
enum InsertDefaultWireKind {
    #[serde(rename = "default_v1")]
    Default,
    #[serde(rename = "programmatic_default_v1")]
    ProgrammaticDefault,
}

/// A format-generic, externally tagged compatibility mirror of every serializable `SqlValue`
/// arm, plus the one reserved DEFAULT marker. This intentionally avoids `#[serde(untagged)]`:
/// untagged buffering cannot faithfully replay every scalar representation (notably `i128`
/// NUMERIC payloads) to a nested enum deserializer.
///
/// Keep the scalar variants in `SqlValue` declaration order. That retains legacy binary enum
/// discriminants as well as the established JSON objects/strings. `Parameter` has no legacy wire
/// form (`SqlValue` skips it and `InsertCell` rejects it during serialization), so it is not a
/// compatibility arm here.
#[derive(serde::Serialize, serde::Deserialize)]
enum InsertCellWire {
    Null,
    Int4(i32),
    Int8(i64),
    Numeric(super::Decimal128),
    Bool(bool),
    Text(String),
    Date(i32),
    Timestamp(i64),
    Uuid([u8; 16]),
    Int2(i16),
    #[serde(rename = "$gpu_db_insert_cell")]
    Default(InsertDefaultWireKind),
}

impl From<InsertCellWire> for InsertCell {
    fn from(wire: InsertCellWire) -> Self {
        match wire {
            InsertCellWire::Null => Self::programmatic(SqlValue::Null),
            InsertCellWire::Int4(value) => Self::programmatic(SqlValue::Int4(value)),
            InsertCellWire::Int8(value) => Self::programmatic(SqlValue::Int8(value)),
            InsertCellWire::Numeric(value) => Self::programmatic(SqlValue::Numeric(value)),
            InsertCellWire::Bool(value) => Self::programmatic(SqlValue::Bool(value)),
            InsertCellWire::Text(value) => Self::programmatic(SqlValue::Text(value)),
            InsertCellWire::Date(value) => Self::programmatic(SqlValue::Date(value)),
            InsertCellWire::Timestamp(value) => Self::programmatic(SqlValue::Timestamp(value)),
            InsertCellWire::Uuid(value) => Self::programmatic(SqlValue::Uuid(value)),
            InsertCellWire::Int2(value) => Self::programmatic(SqlValue::Int2(value)),
            InsertCellWire::Default(InsertDefaultWireKind::Default) => Self::sql_default(),
            InsertCellWire::Default(InsertDefaultWireKind::ProgrammaticDefault) => {
                Self::programmatic_default()
            }
        }
    }
}

impl serde::Serialize for InsertCell {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Value {
                value: SqlValue::Parameter { .. },
                ..
            } => Err(serde::ser::Error::custom(
                "unbound INSERT parameter cannot be serialized",
            )),
            // Preserve pre-INSERT-001 typed-command byte shape for ordinary value rows. The
            // value's in-memory provenance is intentionally not a durable SQL-AST contract.
            Self::Value { value, .. } => serde::Serialize::serialize(value, serializer),
            Self::Default { provenance } => serde::Serialize::serialize(
                &InsertCellWire::Default(match provenance {
                    InsertDefaultProvenance::SqlKeyword => InsertDefaultWireKind::Default,
                    InsertDefaultProvenance::Programmatic => {
                        InsertDefaultWireKind::ProgrammaticDefault
                    }
                }),
                serializer,
            ),
        }
    }
}

impl<'de> serde::Deserialize<'de> for InsertCell {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <InsertCellWire as serde::Deserialize>::deserialize(deserializer).map(Into::into)
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Insert {
    pub table: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<InsertCell>>,
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
