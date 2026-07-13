use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use std::sync::{Mutex, OnceLock};

#[cfg(test)]
use gpu_db_protocol::FrontendMessage;
#[cfg(test)]
use gpu_db_protocol::{
    is_copy_statement, parse_copy_from_stdin, parse_copy_to_stdout_table, CopyOptions,
};
use gpu_db_protocol::{
    parse_command, AclRelationKind, ColumnDefault, Command, CopyFormat, CopyParseError,
    DatabasePrivilege, FunctionPrivilege, ParseError, PublicationTarget, SchemaPrivilege,
    SelectFilter, SelectFilterOp, SelectProjection, SqlValue, TablePrivilege, TablespacePrivilege,
    SUPPORTED_SQL_TYPES,
};
use gpu_db_protocol::{DescribeTarget, SqlType};

#[path = "gpu-db-server/frontend_transport.rs"]
mod frontend_transport;
use frontend_transport::ReadWrite;
#[path = "gpu-db-server/frontend_dispatch.rs"]
mod frontend_dispatch;
use frontend_dispatch::handle_ready_client;
#[cfg(test)]
use frontend_dispatch::{
    test_handle_frontend_message as handle_frontend_message,
    test_unsupported_frontend_message as unsupported_frontend_message,
};
#[path = "gpu-db-server/extended_query.rs"]
mod extended_query;
#[cfg(test)]
use extended_query::test_execute_portal_batch as execute_portal_batch;
use extended_query::{handle_bind, handle_close, handle_describe, handle_execute, handle_parse};
#[path = "gpu-db-server/sql_execute_syntax.rs"]
mod sql_execute_syntax;
use sql_execute_syntax::{parse_sql_execute, strip_leading_sql_comments, strip_sql_comments};
#[path = "gpu-db-server/sql_prepare.rs"]
mod sql_prepare;
use sql_prepare::{
    describe_extended_query_columns, execute_sql_prepared_result, sql_execute_describe_error,
    try_execute_sql_prepared_statement,
};
#[cfg(test)]
use sql_prepare::{
    parse_sql_deallocate, parse_sql_prepare, parse_sql_prepare_name, SqlDeallocateTarget,
};
#[path = "gpu-db-server/cursor.rs"]
mod cursor;
use cursor::{
    execute_declare_cursor, is_unsupported_declare_cursor_statement, parse_declare_cursor,
    try_execute_cursor_statement,
};
#[cfg(test)]
use cursor::{
    execute_fetch_forward, execute_move_forward, is_unsupported_fetch_cursor_statement,
    is_unsupported_move_cursor_statement, parse_close_cursor, parse_fetch_forward,
    parse_move_forward, CloseCursorTarget,
};
#[path = "gpu-db-server/extended_dml.rs"]
mod extended_dml;
use extended_dml::{execute_extended_delete, execute_extended_insert, execute_extended_update};
#[path = "gpu-db-server/copy_execution.rs"]
mod copy_execution;
use copy_execution::{
    apply_copy_in_rows, begin_copy_from_stdin, execute_copy_to_stdout, handle_copy_data,
    try_execute_copy_statement,
};
#[path = "gpu-db-server/ddl_syntax.rs"]
mod ddl_syntax;
use ddl_syntax::unsupported_foreign_key_option_query;
#[cfg(test)]
use ddl_syntax::{
    parse_alter_table_drop_constraint, parse_drop_table, parse_truncate_table, DropConstraint,
    DropTable, ParsedTruncateTable,
};
#[path = "gpu-db-server/ddl_execution.rs"]
mod ddl_execution;
#[cfg(test)]
use ddl_execution::rename_table_in_session;
use ddl_execution::{execute_parsed_table_ddl, try_execute_ddl_statement};
#[path = "gpu-db-server/session_compat.rs"]
mod session_compat;
use session_compat::{
    execute_session_compat_fallback, try_execute_session_compat_query,
    try_execute_session_control_statement,
};
#[path = "gpu-db-server/pg_dump_compat.rs"]
mod pg_dump_compat;
use pg_dump_compat::try_execute_pg_dump_compat_statement;
#[path = "gpu-db-server/session_commands.rs"]
mod session_commands;
use session_commands::try_execute_session_command;
#[path = "gpu-db-server/bootstrap_ddl.rs"]
mod bootstrap_ddl;
use bootstrap_ddl::try_execute_bootstrap_ddl;
#[path = "gpu-db-server/cluster_ddl.rs"]
mod cluster_ddl;
use cluster_ddl::execute_cluster_ddl;
#[path = "gpu-db-server/index_ddl.rs"]
mod index_ddl;
use index_ddl::execute_index_ddl;
#[cfg(test)]
use index_ddl::rename_index_in_session;
#[path = "gpu-db-server/view_ddl.rs"]
mod view_ddl;
use view_ddl::{execute_view_ddl, try_execute_view_catalog_query};
#[cfg(test)]
use view_ddl::{
    test_psql_describe_materialized_views_catalog_query as psql_describe_materialized_views_catalog_query,
    test_psql_describe_materialized_views_verbose_catalog_query as psql_describe_materialized_views_verbose_catalog_query,
    test_psql_describe_views_catalog_query as psql_describe_views_catalog_query,
    test_psql_describe_views_verbose_catalog_query as psql_describe_views_verbose_catalog_query,
};
#[path = "gpu-db-server/function_execution.rs"]
mod function_execution;
use function_execution::execute_function_command;
#[cfg(test)]
use function_execution::{execute_function_result, rename_function_in_session};
#[path = "gpu-db-server/sequence_execution.rs"]
mod sequence_execution;
use sequence_execution::{
    create_implicit_sequence, execute_sequence_command, next_sequence_value, sequence_target_error,
    try_execute_sequence_catalog_query,
};
#[cfg(test)]
use sequence_execution::{
    rename_sequence_in_session,
    test_psql_describe_sequence_verbose_rows as psql_describe_sequence_verbose_rows,
    test_psql_describe_sequences_catalog_query as psql_describe_sequences_catalog_query,
    test_psql_describe_sequences_verbose_catalog_query as psql_describe_sequences_verbose_catalog_query,
};
#[path = "gpu-db-server/domain_ddl.rs"]
mod domain_ddl;
use domain_ddl::{execute_domain_ddl, try_execute_domain_catalog_query};
#[cfg(test)]
use domain_ddl::{
    test_psql_list_domains_catalog_query as psql_list_domains_catalog_query,
    test_psql_list_domains_verbose_catalog_query as psql_list_domains_verbose_catalog_query,
};
#[path = "gpu-db-server/replication_catalog.rs"]
mod replication_catalog;
use replication_catalog::{
    execute_replication_catalog_command, try_execute_replication_catalog_query,
};
#[cfg(test)]
use replication_catalog::{
    test_catalog_describe_publication_query_oid as catalog_describe_publication_query_oid,
    test_psql_describe_schema_publications_query as psql_describe_schema_publications_query,
    test_psql_list_publications_catalog_query as psql_list_publications_catalog_query,
    test_psql_list_publications_verbose_catalog_query as psql_list_publications_verbose_catalog_query,
    test_psql_list_subscriptions_catalog_query as psql_list_subscriptions_catalog_query,
};
#[path = "gpu-db-server/role_ddl.rs"]
mod role_ddl;
use role_ddl::execute_role_ddl;
#[path = "gpu-db-server/catalog_comments.rs"]
mod catalog_comments;
use catalog_comments::execute_catalog_comment;
#[path = "gpu-db-server/acl_execution.rs"]
mod acl_execution;
use acl_execution::{
    database_exists, execute_acl_command, function_access_permission_error,
    object_access_permission_error, role_exists, role_has_dependencies, schema_permission_error,
    schema_usage_permission_error, tablespace_exists,
};
#[cfg(test)]
use acl_execution::{
    grant_default_table_acl, grant_function_acl, grant_relation_acl, grant_schema_acl,
    revoke_relation_acl,
};
#[path = "gpu-db-server/simple_dml.rs"]
mod simple_dml;
use simple_dml::execute_simple_dml;
#[path = "gpu-db-server/select_execution.rs"]
mod select_execution;
#[cfg(test)]
use select_execution::row_matches_select_filters;
use select_execution::{
    execute_select_result, execute_simple_select, format_sql_value, materialize_select_rows,
    row_matches_delete_filters, select_filter_matches,
};
#[path = "gpu-db-server/integrity_validation.rs"]
mod integrity_validation;
use integrity_validation::{
    validate_check_constraints, validate_foreign_keys, validate_unique_indexes,
};
#[path = "gpu-db-server/table_constraints.rs"]
mod table_constraints;
use table_constraints::{
    add_check_constraint_to_session, add_primary_key_to_session, add_unique_constraint_to_session,
};
#[path = "gpu-db-server/column_defaults.rs"]
mod column_defaults;
use column_defaults::{
    add_column_default_supported, column_default_matches_type, evaluate_column_default,
    format_column_default_expr, preflight_column_default_target, resolve_column_domain_type,
};
#[path = "gpu-db-server/backend_adapter.rs"]
mod backend_adapter;
use backend_adapter::*;
#[path = "gpu-db-server/bind_describe.rs"]
mod bind_describe;
#[path = "gpu-db-server/server_bootstrap.rs"]
mod server_bootstrap;
use bind_describe::{
    bind_parameter_error_field, bind_query_parameters, contains_zero_placeholder,
    describe_query_columns, expected_parameter_count, format_code_count_is_valid,
    is_supported_extended_dml, max_placeholder_index, negative_limit_error_field,
    negative_offset_error_field, resolve_prepared_parameter_type_oids,
    sql_execute_argument_placeholder_index, sql_execute_parameter_error_field,
    sql_execute_parameter_type_mapping_error,
};
#[cfg(test)]
use bind_describe::{
    test_describe_parameterized_select_shape as describe_parameterized_select_shape,
    test_replace_parameter_placeholders_with_dummy_literals as replace_parameter_placeholders_with_dummy_literals,
    test_replace_unquoted_placeholder as replace_unquoted_placeholder,
};

const PUBLIC_NAMESPACE_OID: u32 = 2200;
const POSTGRES_DATABASE_OID: u32 = 5;
const PG_EXTENSION_CLASS_OID: u32 = 3079;
const PG_LANGUAGE_CLASS_OID: u32 = 2612;
const PG_PUBLICATION_CLASS_OID: u32 = 6104;
const PG_SUBSCRIPTION_CLASS_OID: u32 = 6100;
const PLPGSQL_EXTENSION_OID: u32 = 13_500;
const PLPGSQL_LANGUAGE_OID: u32 = 13_501;
const PLPGSQL_CALL_HANDLER_OID: u32 = 13_502;
const PLPGSQL_INLINE_HANDLER_OID: u32 = 13_503;
const PLPGSQL_VALIDATOR_OID: u32 = 13_504;
const PLPGSQL_DESCRIPTION: &str = "PL/pgSQL procedural language";
const FIRST_USER_INDEX_OID: u32 = 20_000;
static SHARED_CATALOG: OnceLock<Mutex<SharedCatalog>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq)]
struct Column {
    name: String,
    oid: u32,
    type_size: i16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ErrorField {
    code: &'static str,
    message: &'static str,
    position: Option<&'static str>,
}

fn text_column(name: &str) -> Column {
    Column {
        name: name.to_string(),
        oid: gpu_db_protocol::SqlType::Text.postgres_oid(),
        type_size: gpu_db_protocol::SqlType::Text.type_size(),
    }
}

fn int4_column(name: &str) -> Column {
    Column {
        name: name.to_string(),
        oid: gpu_db_protocol::SqlType::Int4.postgres_oid(),
        type_size: gpu_db_protocol::SqlType::Int4.type_size(),
    }
}

fn int8_column(name: &str) -> Column {
    Column {
        name: name.to_string(),
        oid: 20,
        type_size: 8,
    }
}

fn numeric_column(name: &str) -> Column {
    Column {
        name: name.to_string(),
        oid: 1700,
        type_size: -1,
    }
}

fn bool_column(name: &str) -> Column {
    Column {
        name: name.to_string(),
        oid: 16,
        type_size: 1,
    }
}

/// Result-set [`Column`] metadata for a declared column type. Maps each storable
/// `SqlType` to its wire OID/size via the per-type helpers above.
fn column_for_sql_type(ty: gpu_db_protocol::SqlType, name: &str) -> Column {
    match ty {
        gpu_db_protocol::SqlType::Int2 => Column {
            name: name.to_string(),
            oid: gpu_db_protocol::SqlType::Int2.postgres_oid(),
            type_size: gpu_db_protocol::SqlType::Int2.type_size(),
        },
        gpu_db_protocol::SqlType::Int4 => int4_column(name),
        gpu_db_protocol::SqlType::Int8 => int8_column(name),
        gpu_db_protocol::SqlType::Numeric { .. } => numeric_column(name),
        gpu_db_protocol::SqlType::Bool => bool_column(name),
        gpu_db_protocol::SqlType::Text => text_column(name),
        gpu_db_protocol::SqlType::Date => Column {
            name: name.to_string(),
            oid: gpu_db_protocol::SqlType::Date.postgres_oid(),
            type_size: gpu_db_protocol::SqlType::Date.type_size(),
        },
        gpu_db_protocol::SqlType::Timestamp => Column {
            name: name.to_string(),
            oid: gpu_db_protocol::SqlType::Timestamp.postgres_oid(),
            type_size: gpu_db_protocol::SqlType::Timestamp.type_size(),
        },
        gpu_db_protocol::SqlType::Uuid => Column {
            name: name.to_string(),
            oid: gpu_db_protocol::SqlType::Uuid.postgres_oid(),
            type_size: gpu_db_protocol::SqlType::Uuid.type_size(),
        },
    }
}

fn copy_parse_error_field(error: CopyParseError) -> ErrorField {
    ErrorField {
        code: error.postgres_code(),
        message: error.postgres_message(),
        position: None,
    }
}

fn sql_value_matches_type(value: &SqlValue, ty: gpu_db_protocol::SqlType) -> bool {
    matches!(
        (value, ty),
        (SqlValue::Int4(_), gpu_db_protocol::SqlType::Int4)
            | (SqlValue::Text(_), gpu_db_protocol::SqlType::Text)
    )
}

fn index_definition_prefix(unique: bool) -> &'static str {
    if unique {
        "CREATE UNIQUE INDEX"
    } else {
        "CREATE INDEX"
    }
}

fn catalog_index_definition(index: &CatalogIndex) -> String {
    format!(
        "{} {} ON public.{} USING btree ({})",
        index_definition_prefix(index.unique),
        index.name,
        index.table,
        index.column
    )
}

struct Session {
    in_transaction: bool,
    prepared: HashMap<String, PreparedStatement>,
    portals: HashMap<String, Portal>,
    cursors: HashMap<String, Cursor>,
    tables: HashMap<String, Table>,
    views: HashMap<String, View>,
    materialized_views: HashMap<String, MaterializedView>,
    functions: BTreeMap<String, FunctionInfo>,
    sequences: HashMap<String, Sequence>,
    domains: BTreeMap<String, Domain>,
    publications: BTreeMap<String, Publication>,
    subscriptions: BTreeMap<String, Subscription>,
    roles: BTreeMap<String, RoleInfo>,
    databases: BTreeMap<String, DatabaseInfo>,
    tablespaces: BTreeMap<String, TablespaceInfo>,
    current_role: Option<String>,
    public_schema_exists: bool,
    public_schema_implicit: bool,
    currval_sequences: HashMap<String, i64>,
    indexes: Vec<CatalogIndex>,
    table_acls: BTreeMap<String, BTreeMap<String, BTreeSet<TablePrivilege>>>,
    database_acls: BTreeMap<String, BTreeMap<String, BTreeSet<DatabasePrivilege>>>,
    tablespace_acls: BTreeMap<String, BTreeMap<String, BTreeSet<TablespacePrivilege>>>,
    schema_acl: BTreeMap<String, BTreeSet<SchemaPrivilege>>,
    default_table_acl: BTreeMap<String, BTreeSet<TablePrivilege>>,
    comments: BTreeMap<CatalogCommentTarget, String>,
    dirty_tables: BTreeSet<String>,
    dirty_views: BTreeSet<String>,
    dirty_materialized_views: BTreeSet<String>,
    dirty_functions: BTreeSet<String>,
    dirty_sequences: BTreeSet<String>,
    dirty_domains: BTreeSet<String>,
    dirty_publications: BTreeSet<String>,
    dirty_subscriptions: BTreeSet<String>,
    dirty_roles: BTreeSet<String>,
    dirty_databases: BTreeSet<String>,
    dirty_tablespaces: BTreeSet<String>,
    dirty_schema: bool,
    dirty_indexes: bool,
    dirty_table_acls: BTreeSet<String>,
    dirty_database_acls: BTreeSet<String>,
    dirty_tablespace_acls: BTreeSet<String>,
    dirty_schema_acl: bool,
    dirty_default_table_acl: bool,
    dirty_comment_targets: BTreeSet<CatalogCommentTarget>,
    copy_in: Option<CopyInState>,
    next_relation_oid: u32,
    shared_catalog: bool,
}

#[derive(Clone, Debug)]
struct SharedCatalog {
    tables: HashMap<String, Table>,
    views: HashMap<String, View>,
    materialized_views: HashMap<String, MaterializedView>,
    functions: BTreeMap<String, FunctionInfo>,
    sequences: HashMap<String, Sequence>,
    domains: BTreeMap<String, Domain>,
    publications: BTreeMap<String, Publication>,
    subscriptions: BTreeMap<String, Subscription>,
    roles: BTreeMap<String, RoleInfo>,
    databases: BTreeMap<String, DatabaseInfo>,
    tablespaces: BTreeMap<String, TablespaceInfo>,
    public_schema_exists: bool,
    public_schema_implicit: bool,
    indexes: Vec<CatalogIndex>,
    table_acls: BTreeMap<String, BTreeMap<String, BTreeSet<TablePrivilege>>>,
    database_acls: BTreeMap<String, BTreeMap<String, BTreeSet<DatabasePrivilege>>>,
    tablespace_acls: BTreeMap<String, BTreeMap<String, BTreeSet<TablespacePrivilege>>>,
    schema_acl: BTreeMap<String, BTreeSet<SchemaPrivilege>>,
    default_table_acl: BTreeMap<String, BTreeSet<TablePrivilege>>,
    comments: BTreeMap<CatalogCommentTarget, String>,
    next_relation_oid: u32,
}

impl Default for SharedCatalog {
    fn default() -> Self {
        Self {
            tables: HashMap::new(),
            views: HashMap::new(),
            materialized_views: HashMap::new(),
            functions: BTreeMap::new(),
            sequences: HashMap::new(),
            domains: BTreeMap::new(),
            publications: BTreeMap::new(),
            subscriptions: BTreeMap::new(),
            roles: BTreeMap::new(),
            databases: BTreeMap::new(),
            tablespaces: BTreeMap::new(),
            public_schema_exists: true,
            public_schema_implicit: true,
            indexes: Vec::new(),
            table_acls: BTreeMap::new(),
            database_acls: BTreeMap::new(),
            tablespace_acls: BTreeMap::new(),
            schema_acl: BTreeMap::new(),
            default_table_acl: BTreeMap::new(),
            comments: BTreeMap::new(),
            next_relation_oid: FIRST_USER_RELATION_OID,
        }
    }
}

fn shared_catalog() -> &'static Mutex<SharedCatalog> {
    SHARED_CATALOG.get_or_init(|| Mutex::new(SharedCatalog::default()))
}

impl Default for Session {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Session {
    fn new(shared_catalog_enabled: bool) -> Self {
        let catalog = if shared_catalog_enabled {
            shared_catalog()
                .lock()
                .expect("shared catalog mutex poisoned")
                .clone()
        } else {
            SharedCatalog::default()
        };
        Self {
            in_transaction: false,
            prepared: HashMap::new(),
            portals: HashMap::new(),
            cursors: HashMap::new(),
            tables: catalog.tables,
            views: catalog.views,
            materialized_views: catalog.materialized_views,
            functions: catalog.functions,
            sequences: catalog.sequences,
            domains: catalog.domains,
            publications: catalog.publications,
            subscriptions: catalog.subscriptions,
            roles: catalog.roles,
            databases: catalog.databases,
            tablespaces: catalog.tablespaces,
            current_role: None,
            public_schema_exists: catalog.public_schema_exists,
            public_schema_implicit: catalog.public_schema_implicit,
            currval_sequences: HashMap::new(),
            indexes: catalog.indexes,
            table_acls: catalog.table_acls,
            database_acls: catalog.database_acls,
            tablespace_acls: catalog.tablespace_acls,
            schema_acl: catalog.schema_acl,
            default_table_acl: catalog.default_table_acl,
            comments: catalog.comments,
            dirty_tables: BTreeSet::new(),
            dirty_views: BTreeSet::new(),
            dirty_materialized_views: BTreeSet::new(),
            dirty_functions: BTreeSet::new(),
            dirty_sequences: BTreeSet::new(),
            dirty_domains: BTreeSet::new(),
            dirty_publications: BTreeSet::new(),
            dirty_subscriptions: BTreeSet::new(),
            dirty_roles: BTreeSet::new(),
            dirty_databases: BTreeSet::new(),
            dirty_tablespaces: BTreeSet::new(),
            dirty_schema: false,
            dirty_indexes: false,
            dirty_table_acls: BTreeSet::new(),
            dirty_database_acls: BTreeSet::new(),
            dirty_tablespace_acls: BTreeSet::new(),
            dirty_schema_acl: false,
            dirty_default_table_acl: false,
            dirty_comment_targets: BTreeSet::new(),
            copy_in: None,
            next_relation_oid: catalog.next_relation_oid,
            shared_catalog: shared_catalog_enabled,
        }
    }

    fn mark_table_dirty(&mut self, table: impl Into<String>) {
        self.dirty_tables.insert(table.into());
    }

    fn mark_view_dirty(&mut self, view: impl Into<String>) {
        self.dirty_views.insert(view.into());
    }

    fn mark_materialized_view_dirty(&mut self, view: impl Into<String>) {
        self.dirty_materialized_views.insert(view.into());
    }

    fn mark_function_dirty(&mut self, function: impl Into<String>) {
        self.dirty_functions.insert(function.into());
    }

    fn mark_sequence_dirty(&mut self, sequence: impl Into<String>) {
        self.dirty_sequences.insert(sequence.into());
    }

    fn mark_domain_dirty(&mut self, domain: impl Into<String>) {
        self.dirty_domains.insert(domain.into());
    }

    fn mark_publication_dirty(&mut self, publication: impl Into<String>) {
        self.dirty_publications.insert(publication.into());
    }

    fn mark_subscription_dirty(&mut self, subscription: impl Into<String>) {
        self.dirty_subscriptions.insert(subscription.into());
    }

    fn mark_role_dirty(&mut self, role: impl Into<String>) {
        self.dirty_roles.insert(role.into());
    }

    fn mark_database_dirty(&mut self, database: impl Into<String>) {
        self.dirty_databases.insert(database.into());
    }

    fn mark_tablespace_dirty(&mut self, tablespace: impl Into<String>) {
        self.dirty_tablespaces.insert(tablespace.into());
    }

    fn mark_schema_dirty(&mut self) {
        self.dirty_schema = true;
    }

    fn mark_table_acl_dirty(&mut self, table: impl Into<String>) {
        self.dirty_table_acls.insert(table.into());
    }

    fn mark_database_acl_dirty(&mut self, database: impl Into<String>) {
        self.dirty_database_acls.insert(database.into());
    }

    fn mark_tablespace_acl_dirty(&mut self, tablespace: impl Into<String>) {
        self.dirty_tablespace_acls.insert(tablespace.into());
    }

    fn mark_schema_acl_dirty(&mut self) {
        self.dirty_schema_acl = true;
    }

    fn mark_default_table_acl_dirty(&mut self) {
        self.dirty_default_table_acl = true;
    }

    fn mark_comment_dirty(&mut self, target: CatalogCommentTarget) {
        self.dirty_comment_targets.insert(target);
    }

    fn persist_catalog_snapshot(&mut self) {
        if !self.shared_catalog {
            self.dirty_tables.clear();
            self.dirty_views.clear();
            self.dirty_materialized_views.clear();
            self.dirty_functions.clear();
            self.dirty_sequences.clear();
            self.dirty_domains.clear();
            self.dirty_publications.clear();
            self.dirty_subscriptions.clear();
            self.dirty_roles.clear();
            self.dirty_databases.clear();
            self.dirty_tablespaces.clear();
            self.dirty_schema = false;
            self.dirty_table_acls.clear();
            self.dirty_database_acls.clear();
            self.dirty_tablespace_acls.clear();
            self.dirty_schema_acl = false;
            self.dirty_default_table_acl = false;
            self.dirty_comment_targets.clear();
            return;
        }
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        for table_name in &self.dirty_tables {
            if let Some(table) = self.tables.get(table_name) {
                catalog.tables.insert(table_name.clone(), table.clone());
            } else {
                catalog.tables.remove(table_name);
            }
        }
        for view_name in &self.dirty_views {
            if let Some(view) = self.views.get(view_name) {
                catalog.views.insert(view_name.clone(), view.clone());
            } else {
                catalog.views.remove(view_name);
            }
        }
        for view_name in &self.dirty_materialized_views {
            if let Some(view) = self.materialized_views.get(view_name) {
                catalog
                    .materialized_views
                    .insert(view_name.clone(), view.clone());
            } else {
                catalog.materialized_views.remove(view_name);
            }
        }
        for function_name in &self.dirty_functions {
            if let Some(function) = self.functions.get(function_name) {
                catalog
                    .functions
                    .insert(function_name.clone(), function.clone());
            } else {
                catalog.functions.remove(function_name);
            }
        }
        for sequence_name in &self.dirty_sequences {
            if let Some(sequence) = self.sequences.get(sequence_name) {
                catalog
                    .sequences
                    .insert(sequence_name.clone(), sequence.clone());
            } else {
                catalog.sequences.remove(sequence_name);
            }
        }
        for domain_name in &self.dirty_domains {
            if let Some(domain) = self.domains.get(domain_name) {
                catalog.domains.insert(domain_name.clone(), domain.clone());
            } else {
                catalog.domains.remove(domain_name);
            }
        }
        for publication_name in &self.dirty_publications {
            if let Some(publication) = self.publications.get(publication_name) {
                catalog
                    .publications
                    .insert(publication_name.clone(), publication.clone());
            } else {
                catalog.publications.remove(publication_name);
            }
        }
        for subscription_name in &self.dirty_subscriptions {
            if let Some(subscription) = self.subscriptions.get(subscription_name) {
                catalog
                    .subscriptions
                    .insert(subscription_name.clone(), subscription.clone());
            } else {
                catalog.subscriptions.remove(subscription_name);
            }
        }
        for role_name in &self.dirty_roles {
            if let Some(role) = self.roles.get(role_name) {
                catalog.roles.insert(role_name.clone(), role.clone());
            } else {
                catalog.roles.remove(role_name);
            }
        }
        for database_name in &self.dirty_databases {
            if let Some(database) = self.databases.get(database_name) {
                catalog
                    .databases
                    .insert(database_name.clone(), database.clone());
            } else {
                catalog.databases.remove(database_name);
            }
        }
        for tablespace_name in &self.dirty_tablespaces {
            if let Some(tablespace) = self.tablespaces.get(tablespace_name) {
                catalog
                    .tablespaces
                    .insert(tablespace_name.clone(), tablespace.clone());
            } else {
                catalog.tablespaces.remove(tablespace_name);
            }
        }
        for table_name in &self.dirty_table_acls {
            if let Some(acl) = self.table_acls.get(table_name) {
                catalog.table_acls.insert(table_name.clone(), acl.clone());
            } else {
                catalog.table_acls.remove(table_name);
            }
        }
        for database_name in &self.dirty_database_acls {
            if let Some(acl) = self.database_acls.get(database_name) {
                catalog
                    .database_acls
                    .insert(database_name.clone(), acl.clone());
            } else {
                catalog.database_acls.remove(database_name);
            }
        }
        for tablespace_name in &self.dirty_tablespace_acls {
            if let Some(acl) = self.tablespace_acls.get(tablespace_name) {
                catalog
                    .tablespace_acls
                    .insert(tablespace_name.clone(), acl.clone());
            } else {
                catalog.tablespace_acls.remove(tablespace_name);
            }
        }
        if self.dirty_default_table_acl {
            catalog.default_table_acl = self.default_table_acl.clone();
            self.dirty_default_table_acl = false;
        }
        if self.dirty_schema_acl {
            catalog.schema_acl = self.schema_acl.clone();
            self.dirty_schema_acl = false;
        }
        if self.dirty_schema {
            catalog.public_schema_exists = self.public_schema_exists;
            catalog.public_schema_implicit = self.public_schema_implicit;
            self.dirty_schema = false;
        }
        catalog.next_relation_oid = catalog.next_relation_oid.max(self.next_relation_oid);
        if self.dirty_indexes {
            let deleted_tables = self
                .dirty_tables
                .iter()
                .filter(|table_name| !self.tables.contains_key(*table_name))
                .cloned()
                .collect::<BTreeSet<_>>();
            let replaced_tables = self
                .dirty_tables
                .iter()
                .filter(|table_name| self.tables.contains_key(*table_name))
                .cloned()
                .collect::<BTreeSet<_>>();
            catalog.indexes.retain(|index| {
                !deleted_tables.contains(&index.table) && !replaced_tables.contains(&index.table)
            });
            for index in self
                .indexes
                .iter()
                .filter(|index| self.tables.contains_key(&index.table))
            {
                if let Some(existing) = catalog
                    .indexes
                    .iter_mut()
                    .find(|existing| existing.name == index.name)
                {
                    *existing = index.clone();
                } else {
                    catalog.indexes.push(index.clone());
                }
            }
            self.dirty_indexes = false;
        }
        for target in &self.dirty_comment_targets {
            if let Some(comment) = self.comments.get(target) {
                catalog.comments.insert(target.clone(), comment.clone());
            } else {
                catalog.comments.remove(target);
            }
        }
        self.dirty_comment_targets.clear();
        self.dirty_tables.clear();
        self.dirty_views.clear();
        self.dirty_materialized_views.clear();
        self.dirty_functions.clear();
        self.dirty_sequences.clear();
        self.dirty_domains.clear();
        self.dirty_publications.clear();
        self.dirty_subscriptions.clear();
        self.dirty_roles.clear();
        self.dirty_databases.clear();
        self.dirty_tablespaces.clear();
        self.dirty_table_acls.clear();
        self.dirty_database_acls.clear();
        self.dirty_tablespace_acls.clear();
    }

    fn close_extended_target(&mut self, target: DescribeTarget, name: &str) {
        match target {
            DescribeTarget::Statement => {
                if matches!(
                    self.prepared.get(name),
                    Some(PreparedStatement::Extended(_))
                ) {
                    self.prepared.remove(name);
                    self.portals
                        .retain(|_, portal| portal.statement_name != name);
                }
            }
            DescribeTarget::Portal => {
                self.portals.remove(name);
            }
        }
    }

    fn replace_extended_statement(&mut self, name: String, query: PreparedQuery) {
        self.prepared
            .insert(name.clone(), PreparedStatement::Extended(query));
        self.portals
            .retain(|_, portal| portal.statement_name != name);
    }

    fn replace_extended_portal(&mut self, name: String, portal: Portal) {
        self.portals.insert(name, portal);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Table {
    oid: u32,
    name: String,
    columns: Vec<CatalogColumn>,
    rows: Vec<Vec<SqlValue>>,
    check_constraints: Vec<CatalogCheckConstraint>,
    foreign_keys: Vec<CatalogForeignKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoleInfo {
    oid: u32,
    name: String,
    login: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DatabaseInfo {
    oid: u32,
    name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TablespaceInfo {
    oid: u32,
    name: String,
    location: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct View {
    oid: u32,
    name: String,
    query: gpu_db_protocol::Select,
    definition: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MaterializedView {
    oid: u32,
    name: String,
    query: gpu_db_protocol::Select,
    definition: String,
    columns: Vec<CatalogColumn>,
    rows: Vec<Vec<SqlValue>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FunctionInfo {
    oid: u32,
    name: String,
    return_type: SqlType,
    body: String,
    acl: BTreeMap<String, BTreeSet<FunctionPrivilege>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Sequence {
    oid: u32,
    name: String,
    last_value: i64,
    is_called: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Domain {
    oid: u32,
    name: String,
    base_type: gpu_db_protocol::SqlType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Publication {
    oid: u32,
    name: String,
    all_tables: bool,
    tables: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Subscription {
    oid: u32,
    name: String,
    connection: String,
    publications: Vec<String>,
    enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogColumn {
    attnum: i16,
    def: gpu_db_protocol::ColumnDef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogIndex {
    name: String,
    table: String,
    column: String,
    unique: bool,
    primary_key: bool,
    unique_constraint: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogCheckConstraint {
    name: String,
    table: String,
    column: String,
    op: SelectFilterOp,
    value: SqlValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogForeignKey {
    name: String,
    table: String,
    column: String,
    referenced_table: String,
    referenced_column: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CatalogCommentTarget {
    Database { database: String },
    Role { role: String },
    Schema { schema: String },
    Tablespace { tablespace: String },
    Table { table: String },
    Column { table: String, attnum: i16 },
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum PreparedStatement {
    AddTen,
    Extended(PreparedQuery),
    PgDumpFunctionDump,
    Sql(PreparedQuery),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedQuery {
    query: String,
    parameter_type_oids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BindParameterError {
    CountMismatch,
    NullUnsupported,
    InvalidTextRepresentation { oid: u32, value: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Portal {
    statement_name: String,
    query: PreparedQuery,
    parameters: Vec<Option<String>>,
    result_format_codes: Vec<i16>,
    described: bool,
    result: Option<SelectResult>,
    position: usize,
    completed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Cursor {
    columns: Vec<Column>,
    rows: Vec<Vec<Option<String>>>,
    position: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CopyInState {
    table: String,
    columns: Vec<String>,
    format: CopyFormat,
    header: bool,
    delimiter: char,
    quote: char,
    escape: char,
    pending_text: String,
    pending_rows: Vec<Vec<SqlValue>>,
    seen_terminator: bool,
    ready_after_done: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SelectResult {
    columns: Vec<Column>,
    rows: Vec<Vec<Option<String>>>,
}

const FIRST_USER_RELATION_OID: u32 = 16_384;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    server_bootstrap::run()
}

fn run_simple_query(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    query: &str,
) -> io::Result<()> {
    let statements = split_simple_query(query);
    if statements.is_empty() {
        write_empty_query_response(stream)?;
        write_ready_for_query(stream, session.in_transaction)?;
        return Ok(());
    }

    for statement in statements {
        execute_statement(stream, session, statement, true)?;
        if session.copy_in.is_some() {
            return Ok(());
        }
    }

    write_ready_for_query(stream, session.in_transaction)
}

fn split_simple_query(query: &str) -> Vec<&str> {
    let mut statements = Vec::new();
    let mut start = 0;
    let mut in_string = false;
    let mut in_quoted_identifier = false;
    let mut in_line_comment = false;
    let mut block_comment_depth = 0usize;
    let mut previous_char: Option<char> = None;
    let mut chars = query.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
            }
            previous_char = Some(ch);
            continue;
        }
        if block_comment_depth > 0 {
            if previous_char == Some('/') && ch == '*' {
                block_comment_depth = block_comment_depth.saturating_add(1);
                previous_char = None;
                continue;
            }
            if previous_char == Some('*') && ch == '/' {
                block_comment_depth = block_comment_depth.saturating_sub(1);
                previous_char = None;
                continue;
            }
            previous_char = Some(ch);
            continue;
        }
        match ch {
            '"' if in_quoted_identifier => {
                if matches!(chars.peek(), Some((_, '"'))) {
                    chars.next();
                } else {
                    in_quoted_identifier = false;
                }
            }
            '"' if !in_string => in_quoted_identifier = true,
            '\'' if in_string => {
                if matches!(chars.peek(), Some((_, '\''))) {
                    chars.next();
                } else {
                    in_string = false;
                }
            }
            '\'' if !in_quoted_identifier => in_string = true,
            '-' if !in_string
                && !in_quoted_identifier
                && matches!(chars.peek(), Some((_, '-'))) =>
            {
                chars.next();
                in_line_comment = true;
                previous_char = None;
                continue;
            }
            '/' if !in_string
                && !in_quoted_identifier
                && matches!(chars.peek(), Some((_, '*'))) =>
            {
                chars.next();
                block_comment_depth = 1;
                previous_char = None;
                continue;
            }
            ';' if !in_string && !in_quoted_identifier => {
                let statement = query[start..idx].trim();
                if !statement.is_empty() {
                    statements.push(statement);
                }
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
        previous_char = Some(ch);
    }
    let statement = query[start..].trim();
    if !statement.is_empty() {
        statements.push(statement);
    }
    statements
}

fn execute_statement(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    statement: &str,
    include_row_description: bool,
) -> io::Result<()> {
    if strip_sql_comments(statement).trim().is_empty() {
        return write_empty_query_response(stream);
    }
    let canonical = canonical_sql(statement);
    if let Some(result) = try_execute_copy_statement(stream, session, statement) {
        return result;
    }
    if let Some(result) = try_execute_cursor_statement(stream, session, statement) {
        return result;
    }

    if let Some(result) = try_execute_sql_prepared_statement(
        stream,
        session,
        statement,
        &canonical,
        include_row_description,
    ) {
        return result;
    }
    if let Some(result) = try_execute_session_control_statement(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_ddl_statement(stream, session, statement) {
        return result;
    }
    if let Some(result) = try_execute_session_compat_query(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_pg_dump_compat_statement(stream, session, &canonical) {
        return result;
    }
    match parse_command(statement) {
        Err(ParseError::NegativeLimit) => {
            return write_error(stream, &negative_limit_error_field())
        }
        Err(ParseError::NegativeOffset) => {
            return write_error(stream, &negative_offset_error_field());
        }
        Err(_) if unsupported_foreign_key_option_query(&canonical) => {
            return write_error(
                stream,
                &ErrorField {
                    code: "0A000",
                    message: "foreign key options beyond single-column immediate constraints are not supported",
                    position: None,
                },
            );
        }
        Err(_) => {}
        Ok(
            command @ (Command::SetRole { .. }
            | Command::Begin
            | Command::Commit { .. }
            | Command::Rollback { .. }
            | Command::ResetAll),
        ) => {
            if let Some(result) = try_execute_session_command(stream, session, &command, &canonical)
            {
                return result;
            }
        }
        Ok(
            command @ (Command::CreateExtension(_)
            | Command::DropExtension(_)
            | Command::CreateSchema(_)
            | Command::DropSchema(_)),
        ) => {
            return try_execute_bootstrap_ddl(stream, session, &command)
                .expect("bootstrap DDL parse arm must be handled");
        }
        Ok(
            command @ (Command::CreateDatabase(_)
            | Command::DropDatabase(_)
            | Command::RenameDatabase(_)
            | Command::CreateTablespace(_)
            | Command::DropTablespace(_)
            | Command::RenameTablespace(_)),
        ) => return execute_cluster_ddl(stream, session, command),
        Ok(
            command @ (Command::CreateTable(_)
            | Command::AddPrimaryKey(_)
            | Command::AddUniqueConstraint(_)
            | Command::AddCheckConstraint(_)
            | Command::AddForeignKey(_)
            | Command::DropConstraint(_)
            | Command::RenameConstraint(_)
            | Command::RenameTable(_)
            | Command::DropTable(_)
            | Command::AlterColumnDefault(_)
            | Command::AddColumn(_)
            | Command::RenameColumn(_)
            | Command::DropColumn(_)),
        ) => return execute_parsed_table_ddl(stream, session, command),
        Ok(
            command @ (Command::CreateIndex(_) | Command::RenameIndex(_) | Command::DropIndex(_)),
        ) => {
            return execute_index_ddl(stream, session, command);
        }
        Ok(
            command @ (Command::CreateView(_)
            | Command::CreateMaterializedView(_)
            | Command::RefreshMaterializedView(_)
            | Command::RenameView(_)
            | Command::RenameMaterializedView(_)
            | Command::DropView(_)
            | Command::DropMaterializedView(_)),
        ) => return execute_view_ddl(stream, session, command),
        Ok(
            command @ (Command::CreateFunction(_)
            | Command::RenameFunction(_)
            | Command::DropFunction(_)
            | Command::SelectFunction(_)),
        ) => {
            return execute_function_command(stream, session, command, include_row_description);
        }
        Ok(
            command @ (Command::CreateSequence(_)
            | Command::SequenceNextVal(_)
            | Command::SequenceCurrVal(_)
            | Command::SequenceSetVal(_)
            | Command::RenameSequence(_)
            | Command::DropSequence(_)),
        ) => {
            return execute_sequence_command(stream, session, command, include_row_description);
        }
        Ok(command @ (Command::CreateDomain(_) | Command::DropDomain(_))) => {
            return execute_domain_ddl(stream, session, command);
        }
        Ok(
            command @ (Command::CreatePublication(_)
            | Command::DropPublication(_)
            | Command::CreateSubscription(_)
            | Command::DropSubscription(_)),
        ) => return execute_replication_catalog_command(stream, session, command),
        Ok(command @ (Command::CreateRole(_) | Command::DropRole(_) | Command::RenameRole(_))) => {
            return execute_role_ddl(stream, session, command);
        }
        Ok(command @ Command::CommentOn(_)) => {
            return execute_catalog_comment(stream, session, command);
        }
        Ok(
            command @ (Command::GrantTable(_)
            | Command::RevokeTable(_)
            | Command::GrantSchema(_)
            | Command::RevokeSchema(_)
            | Command::GrantDatabase(_)
            | Command::RevokeDatabase(_)
            | Command::GrantTablespace(_)
            | Command::RevokeTablespace(_)
            | Command::GrantFunction(_)
            | Command::RevokeFunction(_)
            | Command::GrantDefaultTablePrivileges(_)
            | Command::RevokeDefaultTablePrivileges(_)),
        ) => return execute_acl_command(stream, session, command),
        Ok(command @ (Command::Insert(_) | Command::Delete(_) | Command::Update(_))) => {
            return execute_simple_dml(stream, session, command);
        }
        Ok(command @ Command::Select(_)) => {
            return execute_simple_select(stream, session, command, include_row_description);
        }
        Ok(command) => match command {
            Command::Flush
            | Command::TruncateTable(_)
            | Command::SetKv { .. }
            | Command::DeleteKv { .. }
            | Command::GetKv { .. } => {}
            Command::SetRole { .. }
            | Command::Begin
            | Command::Commit { .. }
            | Command::Rollback { .. }
            | Command::ResetAll => {
                unreachable!("session commands are routed by the preceding parse arm")
            }
            Command::CreateExtension(_)
            | Command::DropExtension(_)
            | Command::CreateSchema(_)
            | Command::DropSchema(_) => {
                unreachable!("bootstrap DDL commands are routed by the preceding parse arm")
            }
            Command::CreateDatabase(_)
            | Command::DropDatabase(_)
            | Command::RenameDatabase(_)
            | Command::CreateTablespace(_)
            | Command::DropTablespace(_)
            | Command::RenameTablespace(_) => {
                unreachable!("cluster DDL commands are routed by the preceding parse arm")
            }
            Command::CreateTable(_)
            | Command::AddPrimaryKey(_)
            | Command::AddUniqueConstraint(_)
            | Command::AddCheckConstraint(_)
            | Command::AddForeignKey(_)
            | Command::DropConstraint(_)
            | Command::RenameConstraint(_)
            | Command::RenameTable(_)
            | Command::DropTable(_)
            | Command::AlterColumnDefault(_)
            | Command::AddColumn(_)
            | Command::RenameColumn(_)
            | Command::DropColumn(_) => {
                unreachable!("parsed table DDL commands are routed by the preceding parse arm")
            }
            Command::CreateIndex(_) | Command::RenameIndex(_) | Command::DropIndex(_) => {
                unreachable!("index DDL commands are routed by the preceding parse arm")
            }
            Command::CreateView(_)
            | Command::CreateMaterializedView(_)
            | Command::RefreshMaterializedView(_)
            | Command::RenameView(_)
            | Command::RenameMaterializedView(_)
            | Command::DropView(_)
            | Command::DropMaterializedView(_) => {
                unreachable!("view DDL commands are routed by the preceding parse arm")
            }
            Command::CreateFunction(_)
            | Command::RenameFunction(_)
            | Command::DropFunction(_)
            | Command::SelectFunction(_) => {
                unreachable!("function commands are routed by the preceding parse arm")
            }
            Command::CreateSequence(_)
            | Command::SequenceNextVal(_)
            | Command::SequenceCurrVal(_)
            | Command::SequenceSetVal(_)
            | Command::RenameSequence(_)
            | Command::DropSequence(_) => {
                unreachable!("sequence commands are routed by the preceding parse arm")
            }
            Command::CreateDomain(_) | Command::DropDomain(_) => {
                unreachable!("domain DDL commands are routed by the preceding parse arm")
            }
            Command::CreatePublication(_)
            | Command::DropPublication(_)
            | Command::CreateSubscription(_)
            | Command::DropSubscription(_) => {
                unreachable!("replication catalog commands are routed by the preceding parse arm")
            }
            Command::CreateRole(_) | Command::DropRole(_) | Command::RenameRole(_) => {
                unreachable!("role DDL commands are routed by the preceding parse arm")
            }
            Command::CommentOn(_) => {
                unreachable!("catalog-comment commands are routed by the preceding parse arm")
            }
            Command::GrantTable(_)
            | Command::RevokeTable(_)
            | Command::GrantSchema(_)
            | Command::RevokeSchema(_)
            | Command::GrantDatabase(_)
            | Command::RevokeDatabase(_)
            | Command::GrantTablespace(_)
            | Command::RevokeTablespace(_)
            | Command::GrantFunction(_)
            | Command::RevokeFunction(_)
            | Command::GrantDefaultTablePrivileges(_)
            | Command::RevokeDefaultTablePrivileges(_) => {
                unreachable!("ACL commands are routed by the preceding parse arm")
            }
            Command::Insert(_) | Command::Delete(_) | Command::Update(_) => {
                unreachable!("simple-query DML commands are routed by the preceding parse arm")
            }
            Command::Select(_) => {
                unreachable!("simple-query SELECT commands are routed by the preceding parse arm")
            }
        },
    }

    if let Some(rows) = psql_describe_query_type_rows(&canonical) {
        return write_single_row(stream, &[text_column("Column"), text_column("Type")], &rows);
    }
    if canonical
        == "select relname from pg_catalog.pg_class where relnamespace = 'public'::regnamespace and relkind = 'r' order by relname"
    {
        return write_single_row(
            stream,
            &[text_column("relname")],
            &catalog_table_name_rows(session),
        );
    }
    if canonical
        == "select oid, relname from pg_catalog.pg_class where relnamespace = 'public'::regnamespace and relkind = 'r' order by oid"
    {
        return write_single_row(
            stream,
            &[int4_column("oid"), text_column("relname")],
            &catalog_table_oid_rows(session),
        );
    }
    if canonical == psql_describe_tables_catalog_query()
        || canonical == psql_describe_all_schema_tables_catalog_query()
        || canonical == psql_describe_relations_catalog_query()
    {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_psql_describe_table_rows(session),
        );
    }
    if let Some(filter) = psql_describe_tables_catalog_query_filter(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_psql_describe_table_rows_filtered(session, &filter),
        );
    }
    if canonical == psql_describe_tables_verbose_catalog_query()
        || canonical == psql_describe_all_schema_tables_verbose_catalog_query()
    {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_psql_describe_table_verbose_rows(session),
        );
    }
    if let Some(filter) = psql_describe_tables_verbose_catalog_query_filter(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_psql_describe_table_verbose_rows_filtered(session, &filter),
        );
    }
    if let Some(filter) = psql_describe_table_privileges_catalog_query_filter(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Access privileges"),
                text_column("Column privileges"),
                text_column("Policies"),
            ],
            &catalog_psql_describe_table_privilege_rows_filtered(session, &filter),
        );
    }
    if canonical == psql_describe_indexes_verbose_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Table"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            &psql_describe_index_verbose_rows(session),
        );
    }
    if canonical == psql_describe_indexes_catalog_query()
        || psql_describe_indexes_catalog_query_schema_filter(&canonical).is_some()
    {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Table"),
            ],
            &psql_describe_index_rows(session),
        );
    }
    if let Some(result) = try_execute_view_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_sequence_catalog_query(stream, session, &canonical) {
        return result;
    }
    if canonical == psql_describe_functions_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Result data type"),
                text_column("Argument data types"),
                text_column("Type"),
            ],
            &psql_describe_function_rows(session),
        );
    }
    if canonical.starts_with("select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.pg_get_function_result(p.oid) as \"result data type\"")
        && canonical.contains("p.provolatile")
        && canonical.contains("from pg_catalog.pg_proc p")
    {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Result data type"),
                text_column("Argument data types"),
                text_column("Type"),
                text_column("Volatility"),
                text_column("Parallel"),
                text_column("Owner"),
                text_column("Security"),
                text_column("Access privileges"),
                text_column("Language"),
                text_column("Internal name"),
                text_column("Description"),
            ],
            &psql_describe_function_verbose_rows(session),
        );
    }
    if canonical == "select p.oid, n.nspname, p.proname, p.prorettype, pg_catalog.pg_get_function_result(p.oid), p.prosrc from pg_catalog.pg_proc p join pg_catalog.pg_namespace n on n.oid = p.pronamespace where n.nspname = 'public' order by p.proname" {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("proname"),
                int4_column("prorettype"),
                text_column("pg_get_function_result"),
                text_column("prosrc"),
            ],
            &pg_catalog_function_rows(session),
        );
    }
    if canonical == "select p.proname, d.description from pg_catalog.pg_proc p join pg_catalog.pg_namespace n on n.oid = p.pronamespace join pg_catalog.pg_description d on d.objoid = p.oid where n.nspname = 'public' order by p.proname" {
        return write_single_row(
            stream,
            &[text_column("proname"), text_column("description")],
            &pg_catalog_function_description_rows(session),
        );
    }
    if canonical == psql_list_aggregates_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Result data type"),
                text_column("Argument data types"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_conversions_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Source"),
                text_column("Destination"),
                text_column("Default?"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_operators_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Left arg type"),
                text_column("Right arg type"),
                text_column("Result type"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_collations_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Provider"),
                text_column("Collate"),
                text_column("Ctype"),
                text_column("ICU Locale"),
                text_column("ICU Rules"),
                text_column("Deterministic?"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_casts_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Source type"),
                text_column("Target type"),
                text_column("Function"),
                text_column("Implicit?"),
            ],
            &catalog_empty_rows(),
        );
    }
    if let Some(result) = try_execute_replication_catalog_query(stream, session, &canonical) {
        return result;
    }
    if canonical == psql_list_default_access_privileges_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Owner"),
                text_column("Schema"),
                text_column("Type"),
                text_column("Access privileges"),
            ],
            &catalog_psql_default_access_privilege_rows(session),
        );
    }
    if canonical == psql_list_extensions_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Version"),
                text_column("Schema"),
                text_column("Description"),
            ],
            &catalog_psql_extension_rows(session),
        );
    }
    if canonical
        == "select x.tableoid, x.oid, x.extname, n.nspname, x.extrelocatable, x.extversion, x.extconfig, x.extcondition from pg_extension x join pg_namespace n on n.oid = x.extnamespace"
    {
        return write_single_row(
            stream,
            &[
                int4_column("tableoid"),
                int4_column("oid"),
                text_column("extname"),
                text_column("nspname"),
                bool_column("extrelocatable"),
                text_column("extversion"),
                text_column("extconfig"),
                text_column("extcondition"),
            ],
            &catalog_extension_discovery_rows(),
        );
    }
    if canonical
        == "select tableoid, oid, lanname, lanpltrusted, lanplcallfoid, laninline, lanvalidator, lanacl, acldefault('l', lanowner) as acldefault, lanowner from pg_language where lanispl order by oid"
    {
        return write_single_row(
            stream,
            &pg_language_discovery_columns(),
            &pg_language_discovery_rows(),
        );
    }
    if canonical == psql_list_languages_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                bool_column("Trusted"),
                text_column("Description"),
            ],
            &catalog_psql_language_rows(),
        );
    }
    if let Some(result) = try_execute_domain_catalog_query(stream, session, &canonical) {
        return result;
    }
    if canonical == psql_describe_roles_catalog_query()
        || canonical == psql_describe_roles_verbose_catalog_query()
    {
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
        return write_single_row(
            stream,
            &columns,
            &catalog_psql_describe_role_rows(session, verbose),
        );
    }
    if canonical == "select oid, rolname from pg_catalog.pg_roles order by 1" {
        return write_single_row(
            stream,
            &[int4_column("oid"), text_column("rolname")],
            &catalog_role_oid_rows(session),
        );
    }
    if canonical == psql_list_databases_catalog_query() {
        return write_single_row(
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
        );
    }
    if canonical == psql_list_databases_verbose_catalog_query() {
        return write_single_row(
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
        );
    }
    if canonical == "select oid, datname from pg_catalog.pg_database order by datname" {
        return write_single_row(
            stream,
            &[int4_column("oid"), text_column("datname")],
            &catalog_database_oid_rows(session),
        );
    }
    if canonical == "select datname, pg_catalog.array_to_string(datacl, e'\\n') as acl from pg_catalog.pg_database order by datname" {
        return write_single_row(
            stream,
            &[text_column("datname"), text_column("acl")],
            &catalog_database_acl_rows(session),
        );
    }
    if is_pg_dumpall_tablespace_metadata_query(&canonical) {
        return write_single_row(
            stream,
            &pg_dumpall_tablespace_metadata_columns(),
            &pg_dumpall_tablespace_metadata_rows(session),
        );
    }
    if canonical == psql_list_tablespaces_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Location"),
            ],
            &catalog_psql_list_tablespace_rows(session, false),
        );
    }
    if canonical == psql_list_tablespaces_verbose_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Location"),
                text_column("Access privileges"),
                text_column("Options"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_psql_list_tablespace_rows(session, true),
        );
    }
    if canonical == "select oid, spcname, pg_catalog.pg_tablespace_location(oid) as location from pg_catalog.pg_tablespace order by spcname" {
        return write_single_row(
            stream,
            &[int4_column("oid"), text_column("spcname"), text_column("location")],
            &catalog_tablespace_oid_rows(session),
        );
    }
    if canonical == "select spcname, pg_catalog.array_to_string(spcacl, e'\\n') as acl from pg_catalog.pg_tablespace order by spcname" {
        return write_single_row(
            stream,
            &[text_column("spcname"), text_column("acl")],
            &catalog_tablespace_acl_rows(session),
        );
    }
    if canonical == psql_list_access_methods_catalog_query() {
        return write_single_row(
            stream,
            &[text_column("Name"), text_column("Type")],
            &catalog_psql_list_access_method_rows(),
        );
    }
    if canonical == psql_describe_schemas_catalog_query() {
        return write_single_row(
            stream,
            &[text_column("Name"), text_column("Owner")],
            &catalog_psql_describe_schema_rows(session),
        );
    }
    if psql_describe_schemas_verbose_catalog_query_public_filter(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Access privileges"),
                text_column("Description"),
            ],
            &catalog_psql_describe_schema_verbose_rows(session),
        );
    }
    if canonical == pg_catalog_schema_description_query() {
        return write_single_row(
            stream,
            &[text_column("nspname"), text_column("description")],
            &pg_catalog_schema_description_rows(session),
        );
    }
    if canonical == pg_catalog_namespace_query() {
        return write_single_row(
            stream,
            &[int4_column("oid"), text_column("nspname")],
            &pg_catalog_namespace_rows(session),
        );
    }
    if canonical == pg_catalog_namespace_acl_query() {
        return write_single_row(
            stream,
            &[text_column("nspname"), text_column("nspacl")],
            &pg_catalog_namespace_acl_rows(session),
        );
    }
    if canonical
        == "select n.tableoid, n.oid, n.nspname, n.nspowner, n.nspacl, acldefault('n', n.nspowner) as acldefault from pg_namespace n"
    {
        return write_single_row(
            stream,
            &[
                int4_column("tableoid"),
                int4_column("oid"),
                text_column("nspname"),
                int4_column("nspowner"),
                text_column("nspacl"),
                text_column("acldefault"),
            ],
            &[
                vec![
                    Some("2615".to_string()),
                    Some("11".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("10".to_string()),
                    None,
                    None,
                ],
                vec![
                    Some("2615".to_string()),
                    Some(PUBLIC_NAMESPACE_OID.to_string()),
                    Some("public".to_string()),
                    Some("10".to_string()),
                    schema_acl_array_display(session),
                    Some("{postgres=UC/postgres,=U/postgres}".to_string()),
                ],
            ],
        );
    }
    if let Some(type_name) = psql_describe_type_catalog_query_type(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_rows(&type_name),
        );
    }
    if canonical == psql_describe_pg_catalog_types_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_rows_for_supported_types(),
        );
    }
    if canonical == psql_describe_pg_catalog_types_verbose_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Internal name"),
                text_column("Size"),
                text_column("Elements"),
                text_column("Owner"),
                text_column("Access privileges"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_verbose_rows_for_supported_types(),
        );
    }
    if is_pg_dump_public_namespace_oid_lookup_query(&canonical) {
        return write_single_row(
            stream,
            &[int4_column("oid")],
            &[vec![Some(PUBLIC_NAMESPACE_OID.to_string())]],
        );
    }
    if let Some(table) = pg_dump_table_oid_lookup_query_table(&canonical) {
        return write_single_row(
            stream,
            &[int4_column("oid")],
            &pg_dump_table_oid_lookup_rows(session, &table),
        );
    }
    if is_pg_dump_class_metadata_query(&canonical) {
        return write_single_row(
            stream,
            &pg_dump_class_metadata_columns(),
            &pg_dump_class_metadata_rows(session),
        );
    }
    if canonical == pg_dump_type_metadata_query() {
        return write_single_row(
            stream,
            &pg_dump_type_metadata_columns(),
            &pg_dump_type_metadata_rows(session),
        );
    }
    if canonical == pg_dump_database_metadata_query() {
        return write_single_row(
            stream,
            &pg_dump_database_metadata_columns(),
            &pg_dump_database_metadata_rows(session),
        );
    }
    if is_pg_dump_index_metadata_query(&canonical) {
        return write_single_row(
            stream,
            &pg_dump_index_metadata_columns(),
            &pg_dump_index_metadata_rows(session),
        );
    }
    if is_catalog_foreign_key_metadata_query(&canonical) {
        return write_single_row(
            stream,
            &catalog_foreign_key_metadata_columns(),
            &catalog_foreign_key_metadata_rows(session),
        );
    }
    if let Some(view_oid) = pg_dump_view_definition_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[text_column("viewdef")],
            &pg_dump_view_definition_rows(session, view_oid),
        );
    }
    if let Some(relation_oids) = pg_dump_attrdef_metadata_query_relation_oids(&canonical) {
        return write_single_row(
            stream,
            &pg_dump_attrdef_metadata_columns(),
            &pg_dump_attrdef_metadata_rows(session, &relation_oids),
        );
    }
    if is_pg_dump_function_metadata_query(&canonical) {
        return write_single_row(
            stream,
            &pg_dump_function_metadata_columns(),
            &pg_dump_function_metadata_rows(session),
        );
    }
    if let Some(columns) = pg_dump_empty_catalog_query_columns(&canonical) {
        return write_single_row(stream, &columns, &catalog_empty_rows());
    }
    if is_pg_dump_default_acl_metadata_query(&canonical) {
        return write_single_row(
            stream,
            &pg_dump_default_acl_metadata_columns(),
            &pg_dump_default_acl_metadata_rows(session),
        );
    }
    if catalog_describe_relation_lookup_query_all_schemas(&canonical)
        || catalog_describe_relation_lookup_query_public_namespace(&canonical)
    {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
            ],
            &catalog_describe_relation_lookup_rows_for_public_namespace(session),
        );
    }
    if let Some(table) = catalog_describe_relation_lookup_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
            ],
            &catalog_describe_relation_lookup_rows(session, &table),
        );
    }
    if let Some(oid) = catalog_describe_relation_flags_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("relchecks"),
                text_column("relkind"),
                text_column("relhasindex"),
                text_column("relhasrules"),
                text_column("relhastriggers"),
                text_column("relrowsecurity"),
                text_column("relforcerowsecurity"),
                text_column("relhasoids"),
                text_column("relispartition"),
                text_column("?column?"),
                int4_column("reltablespace"),
                text_column("case"),
                text_column("relpersistence"),
                text_column("relreplident"),
                text_column("amname"),
            ],
            &catalog_describe_relation_flags_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_verbose_attribute_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("attname"),
                text_column("format_type"),
                text_column("pg_get_expr"),
                text_column("attnotnull"),
                text_column("attcollation"),
                text_column("attidentity"),
                text_column("attgenerated"),
                text_column("attstorage"),
                text_column("attcompression"),
                int4_column("attstattarget"),
                text_column("col_description"),
            ],
            &catalog_describe_verbose_attribute_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_attribute_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("attname"),
                text_column("format_type"),
                text_column("pg_get_expr"),
                text_column("attnotnull"),
                text_column("attcollation"),
                text_column("attidentity"),
                text_column("attgenerated"),
            ],
            &catalog_describe_attribute_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_index_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("relname"),
                text_column("indisprimary"),
                text_column("indisunique"),
                text_column("indisclustered"),
                text_column("indisvalid"),
                text_column("pg_get_indexdef"),
                text_column("pg_get_constraintdef"),
                text_column("contype"),
                text_column("condeferrable"),
                text_column("condeferred"),
                text_column("indisreplident"),
                int4_column("reltablespace"),
            ],
            &catalog_describe_index_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_check_constraints_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[text_column("conname"), text_column("pg_get_constraintdef")],
            &catalog_describe_check_constraint_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_foreign_keys_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                bool_column("sametable"),
                text_column("conname"),
                text_column("condef"),
                text_column("ontable"),
            ],
            &catalog_describe_foreign_key_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_referenced_by_foreign_keys_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("conname"),
                text_column("ontable"),
                text_column("condef"),
            ],
            &catalog_describe_referenced_by_foreign_key_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_trigger_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("tgname"),
                text_column("pg_get_triggerdef"),
                text_column("tgenabled"),
                text_column("tgisinternal"),
                text_column("parent"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if let Some(oid) = catalog_describe_policy_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("polname"),
                text_column("polpermissive"),
                text_column("array_to_string"),
                text_column("pg_get_expr"),
                text_column("pg_get_expr"),
                text_column("cmd"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if let Some(oid) = catalog_describe_statistic_ext_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("stxrelid"),
                text_column("nsp"),
                text_column("stxname"),
                text_column("columns"),
                text_column("ndist_enabled"),
                text_column("deps_enabled"),
                text_column("mcv_enabled"),
                int4_column("stxstattarget"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if let Some(oid) = catalog_describe_inherits_parent_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[text_column("oid")],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if let Some(oid) = catalog_describe_inherits_child_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("oid"),
                text_column("relkind"),
                text_column("inhdetachpending"),
                text_column("pg_get_expr"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if canonical == pg_catalog_tables_query() {
        return write_single_row(
            stream,
            &[
                text_column("schemaname"),
                text_column("tablename"),
                text_column("tableowner"),
            ],
            &pg_catalog_table_rows(session),
        );
    }
    if canonical == pg_catalog_indexes_query() {
        return write_single_row(
            stream,
            &[
                text_column("schemaname"),
                text_column("tablename"),
                text_column("indexname"),
                text_column("indexdef"),
            ],
            &pg_catalog_index_rows(session),
        );
    }
    if canonical == pg_catalog_indexes_without_schema_query() {
        return write_single_row(
            stream,
            &[
                text_column("tablename"),
                text_column("indexname"),
                text_column("indexdef"),
            ],
            &pg_catalog_index_rows_without_schema(session),
        );
    }
    if canonical == pg_catalog_class_plain_tables_query() {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("relpersistence"),
            ],
            &pg_catalog_class_plain_table_rows(session),
        );
    }
    if canonical == pg_catalog_class_sequences_query() {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("relpersistence"),
            ],
            &pg_catalog_class_sequence_rows(session),
        );
    }
    if canonical == pg_catalog_class_materialized_views_query() {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("relpersistence"),
            ],
            &pg_catalog_class_materialized_view_rows(session),
        );
    }
    if let Some(tables) = pg_catalog_class_plain_tables_in_query_tables(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("relpersistence"),
            ],
            &pg_catalog_class_plain_table_rows_for_tables(session, &tables),
        );
    }
    if canonical == information_schema_tables_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
            ],
            &information_schema_table_rows(session),
        );
    }
    if canonical == information_schema_base_table_discovery_query() {
        return write_single_row(
            stream,
            &[text_column("table_schema"), text_column("table_name")],
            &information_schema_base_table_discovery_rows(session),
        );
    }
    if let Some(tables) = information_schema_tables_in_query_tables(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
            ],
            &information_schema_table_rows_for_tables(session, &tables),
        );
    }
    if canonical == information_schema_rich_tables_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
                text_column("self_referencing_column_name"),
                text_column("reference_generation"),
                text_column("user_defined_type_catalog"),
                text_column("user_defined_type_schema"),
                text_column("user_defined_type_name"),
                text_column("is_insertable_into"),
                text_column("is_typed"),
                text_column("commit_action"),
            ],
            &information_schema_rich_table_rows(session),
        );
    }
    if let Some(table) = information_schema_rich_tables_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
                text_column("self_referencing_column_name"),
                text_column("reference_generation"),
                text_column("user_defined_type_catalog"),
                text_column("user_defined_type_schema"),
                text_column("user_defined_type_name"),
                text_column("is_insertable_into"),
                text_column("is_typed"),
                text_column("commit_action"),
            ],
            &information_schema_rich_table_rows_for_table(session, &table),
        );
    }
    if let Some(table) = information_schema_rich_tables_catalog_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
                text_column("self_referencing_column_name"),
                text_column("reference_generation"),
                text_column("user_defined_type_catalog"),
                text_column("user_defined_type_schema"),
                text_column("user_defined_type_name"),
                text_column("is_insertable_into"),
                text_column("is_typed"),
                text_column("commit_action"),
            ],
            &information_schema_rich_table_rows_for_table(session, &table),
        );
    }
    if let Some(table) = information_schema_columns_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_column_rows(session, &table),
        );
    }
    if canonical == information_schema_all_columns_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_all_column_rows(session),
        );
    }
    if canonical == information_schema_column_discovery_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_all_column_rows(session),
        );
    }
    if let Some(tables) = information_schema_columns_in_query_tables(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_column_rows_for_tables(session, &tables),
        );
    }
    if let Some(table) = information_schema_column_details_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("column_name"),
                text_column("data_type"),
                text_column("is_nullable"),
                text_column("column_default"),
            ],
            &information_schema_column_detail_rows(session, &table),
        );
    }
    if let Some(table) = information_schema_column_udt_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("column_name"),
                text_column("data_type"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_column_udt_rows(session, &table),
        );
    }
    if canonical == information_schema_rich_columns_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_rich_column_rows(session),
        );
    }
    if canonical == information_schema_extended_columns_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows(session),
        );
    }
    if let Some(table) = information_schema_extended_columns_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_table(session, &table),
        );
    }
    if let Some(table) = information_schema_extended_columns_catalog_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_table(session, &table),
        );
    }
    if let Some(tables) = information_schema_extended_columns_in_query_tables(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_tables(session, &tables),
        );
    }
    if canonical == information_schema_schemata_query() {
        return write_single_row(
            stream,
            &[text_column("schema_name"), text_column("schema_owner")],
            &information_schema_schemata_rows(session),
        );
    }
    if canonical == information_schema_table_constraints_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("constraint_name"),
                text_column("constraint_type"),
            ],
            &information_schema_table_constraint_rows(session),
        );
    }
    if canonical == information_schema_key_column_usage_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                text_column("constraint_name"),
                int4_column("ordinal_position"),
            ],
            &information_schema_key_column_usage_rows(session),
        );
    }
    if canonical == information_schema_views_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("view_definition"),
                text_column("check_option"),
                text_column("is_updatable"),
                text_column("is_insertable_into"),
                text_column("is_trigger_updatable"),
                text_column("is_trigger_deletable"),
                text_column("is_trigger_insertable"),
            ],
            &information_schema_view_rows(session),
        );
    }
    if canonical == pg_catalog_views_query() {
        return write_single_row(
            stream,
            &[
                text_column("schemaname"),
                text_column("viewname"),
                text_column("viewowner"),
                text_column("definition"),
            ],
            &pg_catalog_view_rows(session),
        );
    }
    if canonical == pg_catalog_constraints_query() {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("conname"),
                text_column("contype"),
            ],
            &pg_catalog_constraint_rows(session),
        );
    }
    if canonical == pg_catalog_attrdefs_query() {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("attname"),
                text_column("default_expr"),
            ],
            &pg_catalog_attrdef_rows(session),
        );
    }
    if canonical == pg_catalog_descriptions_query()
        || canonical == pg_catalog_descriptions_with_sequences_query()
        || canonical == pg_catalog_descriptions_with_materialized_views_query()
    {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("attname"),
                text_column("description"),
            ],
            &pg_catalog_description_rows(session),
        );
    }
    if canonical == pg_catalog_table_descriptions_query() {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("attname"),
                text_column("description"),
            ],
            &pg_catalog_table_description_rows(session),
        );
    }
    if canonical == pg_catalog_constraint_descriptions_query() {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("conname"),
                text_column("description"),
            ],
            &pg_catalog_constraint_description_rows(session),
        );
    }
    if canonical == pg_catalog_table_index_descriptions_query()
        || canonical == pg_catalog_table_index_sequence_descriptions_query()
        || canonical == pg_catalog_table_index_sequence_matview_descriptions_query()
    {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("attname"),
                text_column("description"),
            ],
            &pg_catalog_table_index_description_rows(session),
        );
    }
    if canonical == pg_catalog_table_index_descriptions_without_views_query() {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("attname"),
                text_column("description"),
            ],
            &pg_catalog_table_index_description_rows_without_views(session),
        );
    }
    if psql_list_object_descriptions_query(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Object"),
                text_column("Description"),
            ],
            &psql_object_description_rows(session),
        );
    }
    if canonical
        == "select oid, typname, typlen from pg_catalog.pg_type where oid in (23, 25) order by oid"
    {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("typname"),
                int4_column("typlen"),
            ],
            &catalog_type_rows_by_oid_in(&[23, 25]),
        );
    }
    if canonical
        == "select typname, oid, typlen from pg_catalog.pg_type where typname in ('int4', 'text') order by typname"
    {
        return write_single_row(
            stream,
            &[
                text_column("typname"),
                int4_column("oid"),
                int4_column("typlen"),
            ],
            &catalog_type_rows_by_name_in(&["int4", "text"]),
        );
    }
    if canonical
        == "select oid, * from pg_catalog.pg_type where typname in ('hstore','geometry','vector')"
    {
        return write_single_row(stream, &[int4_column("oid")], &[]);
    }
    if let Some(table) = catalog_attribute_query_table(&canonical) {
        let Some(rows) = catalog_attribute_rows(session, &table) else {
            return write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "relation does not exist",
                    position: None,
                },
            );
        };
        return write_single_row(
            stream,
            &[text_column("attname"), int4_column("atttypid")],
            &rows,
        );
    }
    if let Some(table) = catalog_attribute_detail_query_table(&canonical) {
        let Some(rows) = catalog_attribute_detail_rows(session, &table) else {
            return write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "relation does not exist",
                    position: None,
                },
            );
        };
        return write_single_row(
            stream,
            &[
                int4_column("attnum"),
                text_column("attname"),
                int4_column("atttypid"),
                int4_column("attlen"),
            ],
            &rows,
        );
    }
    if let Some(table) = pg_catalog_class_attribute_type_query_table(&canonical) {
        let Some(rows) = pg_catalog_class_attribute_type_rows(session, &table) else {
            return write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "relation does not exist",
                    position: None,
                },
            );
        };
        return write_single_row(
            stream,
            &[
                int4_column("attnum"),
                text_column("attname"),
                text_column("data_type"),
                text_column("attnotnull"),
            ],
            &rows,
        );
    }
    execute_session_compat_fallback(stream, session, &canonical)
}

fn catalog_table_name_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| vec![Some(table.name.clone())])
        .collect::<Vec<_>>()
}

fn catalog_table_oid_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .tables
        .iter()
        .map(|(name, table)| (table.oid, name))
        .collect::<Vec<_>>();
    rows.sort_by_key(|(oid, _)| *oid);
    rows.into_iter()
        .map(|(oid, name)| vec![Some(oid.to_string()), Some(name.clone())])
        .collect()
}

fn psql_describe_tables_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_all_schema_tables_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') order by 1,2"
}

fn psql_describe_relations_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','v','m','s','f','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_tables_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_all_schema_tables_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') order by 1,2"
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PsqlDescribeTablesFilter {
    namespace: String,
    relname_pattern: Option<String>,
}

fn psql_describe_tables_catalog_query_filter(canonical: &str) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and ";
    let namespace_prefix = "n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1,2";
    let rest = canonical.strip_prefix(prefix)?;
    if let Some(namespace) = rest
        .strip_prefix(namespace_prefix)
        .and_then(|rest| rest.strip_suffix(namespace_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: namespace.to_string(),
            relname_pattern: None,
        });
    }

    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }
    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn psql_describe_tables_verbose_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and ";
    let namespace_prefix = "n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1,2";
    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let rest = canonical.strip_prefix(prefix)?;

    if let Some(namespace) = rest
        .strip_prefix(namespace_prefix)
        .and_then(|rest| rest.strip_suffix(namespace_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: namespace.to_string(),
            relname_pattern: None,
        });
    }

    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }

    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn psql_describe_table_privileges_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 's' then 'sequence' when 'f' then 'foreign table' when 'p' then 'partitioned table' end as \"type\", pg_catalog.array_to_string(c.relacl, e'\\n') as \"access privileges\", pg_catalog.array_to_string(array( select attname || e':\\n ' || pg_catalog.array_to_string(attacl, e'\\n ') from pg_catalog.pg_attribute a where attrelid = c.oid and not attisdropped and attacl is not null ), e'\\n') as \"column privileges\", pg_catalog.array_to_string(array( select polname || case when not polpermissive then e' (restrictive)' else '' end || case when polcmd != '*' then e' (' || polcmd::pg_catalog.text || e'):' else e':' end || case when polqual is not null then e'\\n (u): ' || pg_catalog.pg_get_expr(polqual, polrelid) else e'' end || case when polwithcheck is not null then e'\\n (c): ' || pg_catalog.pg_get_expr(polwithcheck, polrelid) else e'' end || case when polroles <> '{0}' then e'\\n to: ' || pg_catalog.array_to_string( array( select rolname from pg_catalog.pg_roles where oid = any (polroles) order by 1 ), e', ') else e'' end from pg_catalog.pg_policy pol where polrelid = c.oid), e'\\n') as \"policies\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('r','v','m','s','f','p') and ";
    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1, 2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1, 2";
    let rest = canonical.strip_prefix(prefix)?;

    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }

    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn psql_describe_indexes_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_indexes_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_indexes_catalog_query_schema_filter(canonical: &str) -> Option<String> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','s','') and n.nspname operator(pg_catalog.~) '^(";
    let suffix = ")$' collate pg_catalog.default order by 1,2";
    let namespace = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    (namespace == "public").then(|| namespace.to_string())
}

fn psql_describe_functions_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.pg_get_function_result(p.oid) as \"result data type\", pg_catalog.pg_get_function_arguments(p.oid) as \"argument data types\", case p.prokind when 'a' then 'agg' when 'w' then 'window' when 'p' then 'proc' else 'func' end as \"type\" from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n on n.oid = p.pronamespace where pg_catalog.pg_function_is_visible(p.oid) and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' order by 1, 2, 4"
}

fn psql_list_aggregates_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.format_type(p.prorettype, null) as \"result data type\", case when p.pronargs = 0 then cast('*' as pg_catalog.text) else pg_catalog.pg_get_function_arguments(p.oid) end as \"argument data types\", pg_catalog.obj_description(p.oid, 'pg_proc') as \"description\" from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n on n.oid = p.pronamespace where p.prokind = 'a' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_function_is_visible(p.oid) order by 1, 2, 4"
}

fn psql_list_conversions_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.conname as \"name\", pg_catalog.pg_encoding_to_char(c.conforencoding) as \"source\", pg_catalog.pg_encoding_to_char(c.contoencoding) as \"destination\", case when c.condefault then 'yes' else 'no' end as \"default?\" from pg_catalog.pg_conversion c join pg_catalog.pg_namespace n on n.oid = c.connamespace where true and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_conversion_is_visible(c.oid) order by 1, 2"
}

fn psql_list_operators_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", o.oprname as \"name\", case when o.oprkind='l' then null else pg_catalog.format_type(o.oprleft, null) end as \"left arg type\", case when o.oprkind='r' then null else pg_catalog.format_type(o.oprright, null) end as \"right arg type\", pg_catalog.format_type(o.oprresult, null) as \"result type\", coalesce(pg_catalog.obj_description(o.oid, 'pg_operator'), pg_catalog.obj_description(o.oprcode, 'pg_proc')) as \"description\" from pg_catalog.pg_operator o left join pg_catalog.pg_namespace n on n.oid = o.oprnamespace where n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_operator_is_visible(o.oid) order by 1, 2, 3, 4"
}

fn psql_list_collations_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.collname as \"name\", case c.collprovider when 'd' then 'default' when 'c' then 'libc' when 'i' then 'icu' end as \"provider\", c.collcollate as \"collate\", c.collctype as \"ctype\", c.colliculocale as \"icu locale\", c.collicurules as \"icu rules\", case when c.collisdeterministic then 'yes' else 'no' end as \"deterministic?\" from pg_catalog.pg_collation c, pg_catalog.pg_namespace n where n.oid = c.collnamespace and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and c.collencoding in (-1, pg_catalog.pg_char_to_encoding(pg_catalog.getdatabaseencoding())) and pg_catalog.pg_collation_is_visible(c.oid) order by 1, 2"
}

fn psql_list_casts_catalog_query() -> &'static str {
    "select pg_catalog.format_type(castsource, null) as \"source type\", pg_catalog.format_type(casttarget, null) as \"target type\", case when c.castmethod = 'b' then '(binary coercible)' when c.castmethod = 'i' then '(with inout)' else p.proname end as \"function\", case when c.castcontext = 'e' then 'no' when c.castcontext = 'a' then 'in assignment' else 'yes' end as \"implicit?\" from pg_catalog.pg_cast c left join pg_catalog.pg_proc p on c.castfunc = p.oid left join pg_catalog.pg_type ts on c.castsource = ts.oid left join pg_catalog.pg_namespace ns on ns.oid = ts.typnamespace left join pg_catalog.pg_type tt on c.casttarget = tt.oid left join pg_catalog.pg_namespace nt on nt.oid = tt.typnamespace where ( (true and pg_catalog.pg_type_is_visible(ts.oid) ) or (true and pg_catalog.pg_type_is_visible(tt.oid) ) ) order by 1, 2"
}

fn psql_list_default_access_privileges_catalog_query() -> &'static str {
    "select pg_catalog.pg_get_userbyid(d.defaclrole) as \"owner\", n.nspname as \"schema\", case d.defaclobjtype when 'r' then 'table' when 's' then 'sequence' when 'f' then 'function' when 't' then 'type' when 'n' then 'schema' end as \"type\", pg_catalog.array_to_string(d.defaclacl, e'\\n') as \"access privileges\" from pg_catalog.pg_default_acl d left join pg_catalog.pg_namespace n on n.oid = d.defaclnamespace order by 1, 2, 3"
}

fn psql_list_extensions_catalog_query() -> &'static str {
    "select e.extname as \"name\", e.extversion as \"version\", n.nspname as \"schema\", c.description as \"description\" from pg_catalog.pg_extension e left join pg_catalog.pg_namespace n on n.oid = e.extnamespace left join pg_catalog.pg_description c on c.objoid = e.oid and c.classoid = 'pg_catalog.pg_extension'::pg_catalog.regclass order by 1"
}

fn psql_list_languages_catalog_query() -> &'static str {
    "select l.lanname as \"name\", pg_catalog.pg_get_userbyid(l.lanowner) as \"owner\", l.lanpltrusted as \"trusted\", d.description as \"description\" from pg_catalog.pg_language l left join pg_catalog.pg_description d on d.classoid = l.tableoid and d.objoid = l.oid and d.objsubid = 0 where l.lanplcallfoid != 0 order by 1"
}

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

fn psql_list_tablespaces_catalog_query() -> &'static str {
    "select spcname as \"name\", pg_catalog.pg_get_userbyid(spcowner) as \"owner\", pg_catalog.pg_tablespace_location(oid) as \"location\" from pg_catalog.pg_tablespace order by 1"
}

fn psql_list_tablespaces_verbose_catalog_query() -> &'static str {
    "select spcname as \"name\", pg_catalog.pg_get_userbyid(spcowner) as \"owner\", pg_catalog.pg_tablespace_location(oid) as \"location\", pg_catalog.array_to_string(spcacl, e'\\n') as \"access privileges\", spcoptions as \"options\", pg_catalog.pg_size_pretty(pg_catalog.pg_tablespace_size(oid)) as \"size\", pg_catalog.shobj_description(oid, 'pg_tablespace') as \"description\" from pg_catalog.pg_tablespace order by 1"
}

fn catalog_psql_list_tablespace_rows(session: &Session, verbose: bool) -> Vec<Vec<Option<String>>> {
    let mut spaces = vec![
        TablespaceInfo {
            oid: 1663,
            name: "pg_default".to_string(),
            location: String::new(),
        },
        TablespaceInfo {
            oid: 1664,
            name: "pg_global".to_string(),
            location: String::new(),
        },
    ];
    spaces.extend(session.tablespaces.values().cloned());
    spaces.sort_by_key(|space| space.name.clone());
    let mut rows = Vec::new();
    for space in spaces {
        let mut row = vec![
            Some(space.name.clone()),
            Some("postgres".to_string()),
            Some(space.location.clone()),
        ];
        if verbose {
            row.push(tablespace_acl_display(session, &space.name));
            row.push(None);
            row.push(Some("0 bytes".to_string()));
            row.push(
                session
                    .comments
                    .get(&CatalogCommentTarget::Tablespace {
                        tablespace: space.name.clone(),
                    })
                    .cloned(),
            );
        }
        rows.push(row);
    }
    rows
}

fn catalog_tablespace_oid_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut spaces = vec![
        TablespaceInfo {
            oid: 1663,
            name: "pg_default".to_string(),
            location: String::new(),
        },
        TablespaceInfo {
            oid: 1664,
            name: "pg_global".to_string(),
            location: String::new(),
        },
    ];
    spaces.extend(session.tablespaces.values().cloned());
    spaces.sort_by_key(|space| space.name.clone());
    spaces
        .into_iter()
        .map(|space| {
            vec![
                Some(space.oid.to_string()),
                Some(space.name),
                Some(space.location),
            ]
        })
        .collect()
}

fn catalog_tablespace_acl_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut spaces = vec!["pg_default".to_string(), "pg_global".to_string()];
    spaces.extend(session.tablespaces.values().map(|space| space.name.clone()));
    spaces.sort();
    spaces
        .into_iter()
        .map(|space| vec![Some(space.clone()), tablespace_acl_display(session, &space)])
        .collect()
}

fn pg_dumpall_tablespace_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("oid"),
        text_column("spcname"),
        text_column("spcowner"),
        text_column("pg_tablespace_location"),
        text_column("spcacl"),
        text_column("acldefault"),
        text_column("array_to_string"),
        text_column("shobj_description"),
    ]
}

fn pg_dumpall_tablespace_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut spaces = session.tablespaces.values().cloned().collect::<Vec<_>>();
    spaces.sort_by_key(|space| space.oid);
    spaces
        .into_iter()
        .map(|space| {
            vec![
                Some(space.oid.to_string()),
                Some(space.name.clone()),
                Some("postgres".to_string()),
                Some(space.location.clone()),
                tablespace_acl_array_display(session, &space.name),
                Some("{postgres=C/postgres}".to_string()),
                None,
                session
                    .comments
                    .get(&CatalogCommentTarget::Tablespace {
                        tablespace: space.name,
                    })
                    .cloned(),
            ]
        })
        .collect()
}

fn is_pg_dumpall_tablespace_metadata_query(canonical: &str) -> bool {
    (canonical.contains("from pg_catalog.pg_tablespace")
        || canonical.contains("from pg_tablespace"))
        && canonical.contains("spcacl")
        && canonical.contains("acldefault")
        && canonical.contains("shobj_description")
}

fn psql_list_access_methods_catalog_query() -> &'static str {
    "select amname as \"name\", case amtype when 'i' then 'index' when 't' then 'table' end as \"type\" from pg_catalog.pg_am order by 1"
}

fn catalog_psql_list_access_method_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![Some("heap".to_string()), Some("Table".to_string())]]
}

fn psql_describe_schemas_catalog_query() -> &'static str {
    "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\" from pg_catalog.pg_namespace n where n.nspname !~ '^pg_' and n.nspname <> 'information_schema' order by 1"
}

fn psql_describe_schemas_verbose_catalog_query_public_filter(canonical: &str) -> bool {
    canonical
        == "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\", pg_catalog.array_to_string(n.nspacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(n.oid, 'pg_namespace') as \"description\" from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1"
}

fn psql_describe_type_catalog_query_type(canonical: &str) -> Option<String> {
    let prefix = "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and (t.typname operator(pg_catalog.~) '^(";
    let suffix = ")$' collate pg_catalog.default or pg_catalog.format_type(t.oid, null) operator(pg_catalog.~) '^(";
    let final_suffix = ")$' collate pg_catalog.default) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2";
    let rest = canonical.strip_prefix(prefix)?;
    let (type_name, rest) = rest.split_once(suffix)?;
    let display_name = rest.strip_suffix(final_suffix)?;
    let matched_type = sql_type_by_catalog_or_display_name(type_name)?;
    (sql_type_by_catalog_or_display_name(display_name) == Some(matched_type))
        .then(|| type_name.to_string())
}

fn psql_describe_pg_catalog_types_query() -> &'static str {
    "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
}

fn psql_describe_pg_catalog_types_verbose_query() -> &'static str {
    "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", t.typname as \"internal name\", case when t.typrelid != 0 then cast('tuple' as pg_catalog.text) when t.typlen < 0 then cast('var' as pg_catalog.text) else cast(t.typlen as pg_catalog.text) end as \"size\", pg_catalog.array_to_string( array( select e.enumlabel from pg_catalog.pg_enum e where e.enumtypid = t.oid order by e.enumsortorder ), e'\\n' ) as \"elements\", pg_catalog.pg_get_userbyid(t.typowner) as \"owner\", pg_catalog.array_to_string(t.typacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
}

fn sql_type_by_catalog_or_display_name(name: &str) -> Option<SqlType> {
    SUPPORTED_SQL_TYPES
        .into_iter()
        .find(|ty| ty.catalog_name() == name || sql_type_display_name(*ty) == name)
}

fn catalog_psql_describe_type_rows(type_name: &str) -> Vec<Vec<Option<String>>> {
    let Some(ty) = sql_type_by_catalog_or_display_name(type_name) else {
        return Vec::new();
    };
    vec![vec![
        Some("pg_catalog".to_string()),
        Some(sql_type_display_name(ty).to_string()),
        None,
    ]]
}

fn supported_sql_types_by_display_name() -> Vec<SqlType> {
    let mut types = SUPPORTED_SQL_TYPES.to_vec();
    types.sort_by_key(|ty| sql_type_display_name(*ty));
    types
}

fn catalog_psql_describe_type_rows_for_supported_types() -> Vec<Vec<Option<String>>> {
    supported_sql_types_by_display_name()
        .into_iter()
        .map(|ty| {
            vec![
                Some("pg_catalog".to_string()),
                Some(sql_type_display_name(ty).to_string()),
                None,
            ]
        })
        .collect()
}

fn catalog_psql_describe_type_verbose_rows_for_supported_types() -> Vec<Vec<Option<String>>> {
    supported_sql_types_by_display_name()
        .into_iter()
        .map(|ty| {
            vec![
                Some("pg_catalog".to_string()),
                Some(sql_type_display_name(ty).to_string()),
                Some(ty.catalog_name().to_string()),
                Some(sql_type_psql_size(ty).to_string()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ]
        })
        .collect()
}

fn sql_type_psql_size(ty: SqlType) -> &'static str {
    match ty.type_size() {
        -1 => "var",
        4 => "4",
        _ => "",
    }
}

fn catalog_psql_describe_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_rows_filtered(
        session,
        &PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: None,
        },
    )
}

fn catalog_psql_describe_table_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_verbose_rows_filtered(
        session,
        &PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: None,
        },
    )
}

fn catalog_psql_describe_table_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    if filter.namespace != "public" {
        return Vec::new();
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .filter(|table| {
            filter
                .relname_pattern
                .as_deref()
                .is_none_or(|pattern| psql_relname_pattern_matches(pattern, &table.name))
        })
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("table".to_string()),
                Some("postgres".to_string()),
            ]
        })
        .collect()
}

fn catalog_psql_describe_table_verbose_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    if filter.namespace != "public" {
        return Vec::new();
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .filter(|table| {
            filter
                .relname_pattern
                .as_deref()
                .is_none_or(|pattern| psql_relname_pattern_matches(pattern, &table.name))
        })
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("table".to_string()),
                Some("postgres".to_string()),
                Some("permanent".to_string()),
                Some("heap".to_string()),
                Some(psql_pretty_table_size(table)),
                session
                    .comments
                    .get(&CatalogCommentTarget::Table {
                        table: table.name.clone(),
                    })
                    .cloned(),
            ]
        })
        .collect()
}

fn psql_pretty_table_size(table: &Table) -> String {
    psql_size_pretty(compat_table_heap_size_bytes(table))
}

fn compat_table_heap_size_bytes(table: &Table) -> u64 {
    table
        .rows
        .iter()
        .map(|row| 24 + row.iter().map(compat_sql_value_size_bytes).sum::<u64>())
        .sum()
}

fn compat_sql_value_size_bytes(value: &SqlValue) -> u64 {
    match value {
        // NULL is sent as a `-1` field length (no bytes); 0 here keeps the size estimate honest.
        SqlValue::Null => 0,
        SqlValue::Int2(_) => 2,
        SqlValue::Int4(_) => 4,
        SqlValue::Text(value) => value.len() as u64,
        SqlValue::Int8(_) => 8,
        // Numeric is sent as text on this endpoint; report its rendered text length.
        SqlValue::Numeric(value) => value.to_decimal_string().len() as u64,
        SqlValue::Bool(_) => 1,
        // Date is sent as its ISO text (YYYY-MM-DD) on this endpoint.
        SqlValue::Date(value) => gpu_db_protocol::datetime::format_date(*value).len() as u64,
        SqlValue::Timestamp(value) => {
            gpu_db_protocol::datetime::format_timestamp(*value).len() as u64
        }
        SqlValue::Uuid(_) => 36, // canonical hyphenated text length
    }
}

fn psql_size_pretty(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes < 10 * KB {
        format!("{bytes} bytes")
    } else if bytes < 10 * MB {
        format!("{} kB", bytes / KB)
    } else if bytes < 10 * GB {
        format!("{} MB", bytes / MB)
    } else {
        format!("{} GB", bytes / GB)
    }
}

fn catalog_psql_describe_table_privilege_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    if filter.namespace != "public" {
        return Vec::new();
    }
    let mut rows = Vec::new();
    for (name, kind) in session
        .tables
        .keys()
        .map(|name| (name.as_str(), "table"))
        .chain(session.views.keys().map(|name| (name.as_str(), "view")))
        .chain(
            session
                .materialized_views
                .keys()
                .map(|name| (name.as_str(), "materialized view")),
        )
        .chain(
            session
                .sequences
                .keys()
                .map(|name| (name.as_str(), "sequence")),
        )
    {
        if filter
            .relname_pattern
            .as_deref()
            .is_some_and(|pattern| !psql_relname_pattern_matches(pattern, name))
        {
            continue;
        }
        rows.push(vec![
            Some("public".to_string()),
            Some(name.to_string()),
            Some(kind.to_string()),
            relation_acl_display(session, name),
            None,
            None,
        ]);
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[2].cmp(&right[2])));
    rows
}

fn relation_acl_display(session: &Session, relation: &str) -> Option<String> {
    let acl = session.table_acls.get(relation)?;
    acl_display(acl)
}

fn relation_acl_array_display(session: &Session, relation: &str) -> Option<String> {
    let acl = session.table_acls.get(relation)?;
    let default = if session.sequences.contains_key(relation) {
        "postgres=rwU/postgres"
    } else {
        "postgres=arwdDxt/postgres"
    };
    acl_array_display_with_default(acl, default)
}

fn schema_acl_display(session: &Session) -> Option<String> {
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

fn schema_acl_array_display(session: &Session) -> Option<String> {
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

fn database_acl_display(session: &Session, database: &str) -> Option<String> {
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

fn tablespace_acl_display(session: &Session, tablespace: &str) -> Option<String> {
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

fn tablespace_acl_array_display(session: &Session, tablespace: &str) -> Option<String> {
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

fn function_acl_display(acl: &BTreeMap<String, BTreeSet<FunctionPrivilege>>) -> Option<String> {
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
    (!rows.is_empty()).then(|| rows.join("\n"))
}

fn function_acl_array_display(
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

fn function_privilege_letters(privileges: &BTreeSet<FunctionPrivilege>) -> String {
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

fn acl_display(acl: &BTreeMap<String, BTreeSet<TablePrivilege>>) -> Option<String> {
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

fn bool_text(value: bool) -> String {
    if value { "t" } else { "f" }.to_string()
}

fn psql_relname_pattern_matches(pattern: &str, table_name: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix(".*") {
        return table_name.starts_with(prefix);
    }
    table_name == pattern
}

fn catalog_psql_describe_schema_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
    ]]
}

fn catalog_psql_describe_schema_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
        schema_acl_display(session),
        session
            .comments
            .get(&CatalogCommentTarget::Schema {
                schema: "public".to_string(),
            })
            .cloned(),
    ]]
}

fn pg_catalog_namespace_query() -> &'static str {
    "select oid, nspname from pg_catalog.pg_namespace where nspname = 'public' order by oid"
}

fn pg_catalog_namespace_acl_query() -> &'static str {
    "select n.nspname, n.nspacl from pg_catalog.pg_namespace n where n.nspname = 'public' order by n.nspname"
}

fn pg_catalog_namespace_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some(PUBLIC_NAMESPACE_OID.to_string()),
        Some("public".to_string()),
    ]]
}

fn pg_catalog_namespace_acl_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        schema_acl_display(session),
    ]]
}

fn pg_catalog_schema_description_query() -> &'static str {
    "select n.nspname, pg_catalog.obj_description(n.oid, 'pg_namespace') as description from pg_catalog.pg_namespace n where n.nspname = 'public' order by n.nspname"
}

fn pg_catalog_schema_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        session
            .comments
            .get(&CatalogCommentTarget::Schema {
                schema: "public".to_string(),
            })
            .cloned(),
    ]]
}

fn catalog_describe_relation_lookup_query_table(canonical: &str) -> Option<String> {
    let prefix = "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(";
    let visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 2, 3";
    if let Some(table) = canonical
        .strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(visible_suffix))
    {
        return Some(table.to_string());
    }

    let namespace_middle =
        ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 2, 3";
    let (table, namespace) = canonical
        .strip_prefix(prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(namespace_middle)?;
    (namespace == "public").then(|| table.to_string())
}

fn catalog_describe_relation_lookup_query_public_namespace(canonical: &str) -> bool {
    canonical
        == "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
}

fn catalog_describe_relation_lookup_query_all_schemas(canonical: &str) -> bool {
    canonical
        == "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace order by 2, 3"
}

fn pg_dump_table_oid_lookup_query_table(canonical: &str) -> Option<String> {
    let prefix = "select c.oid from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid operator(pg_catalog.=) c.relnamespace where c.relkind operator(pg_catalog.=) any (array['r', 's', 'v', 'm', 'f', 'p']) and c.relname operator(pg_catalog.~) '^(";
    let suffix = ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid)";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn is_pg_dump_public_namespace_oid_lookup_query(canonical: &str) -> bool {
    canonical
        == "select oid from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default"
}

fn pg_dump_table_oid_lookup_rows(
    session: &Session,
    relname_pattern: &str,
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    for table in tables {
        if psql_relname_pattern_matches(relname_pattern, &table.name) {
            rows.push(vec![Some(table.oid.to_string())]);
        }
    }
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by_key(|view| view.oid);
    for view in views {
        if psql_relname_pattern_matches(relname_pattern, &view.name) {
            rows.push(vec![Some(view.oid.to_string())]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by_key(|view| view.oid);
    for view in materialized_views {
        if psql_relname_pattern_matches(relname_pattern, &view.name) {
            rows.push(vec![Some(view.oid.to_string())]);
        }
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by_key(|sequence| sequence.oid);
    for sequence in sequences {
        if psql_relname_pattern_matches(relname_pattern, &sequence.name) {
            rows.push(vec![Some(sequence.oid.to_string())]);
        }
    }
    rows
}

fn is_pg_dump_class_metadata_query(canonical: &str) -> bool {
    canonical.starts_with("select c.tableoid, c.oid, c.relname, c.relnamespace, c.relkind, c.reltype, c.relowner, c.relchecks, c.relhasindex, c.relhasrules, c.relpages, c.relhastriggers, c.relpersistence, c.reloftype, c.relacl, acldefault(")
        && canonical.contains("from pg_class c left join pg_depend d")
        && canonical.ends_with("where c.relkind in ('r', 's', 'v', 'c', 'm', 'f', 'p') order by c.oid")
}

fn pg_dump_class_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("relname"),
        int4_column("relnamespace"),
        text_column("relkind"),
        int4_column("reltype"),
        int4_column("relowner"),
        int4_column("relchecks"),
        bool_column("relhasindex"),
        bool_column("relhasrules"),
        int4_column("relpages"),
        bool_column("relhastriggers"),
        text_column("relpersistence"),
        int4_column("reloftype"),
        text_column("relacl"),
        text_column("acldefault"),
        int4_column("foreignserver"),
        text_column("relfrozenxid"),
        text_column("tfrozenxid"),
        int4_column("toid"),
        int4_column("toastpages"),
        text_column("toast_reloptions"),
        int4_column("owning_tab"),
        int4_column("owning_col"),
        text_column("reltablespace"),
        bool_column("relhasoids"),
        bool_column("relispopulated"),
        text_column("relreplident"),
        bool_column("relrowsecurity"),
        bool_column("relforcerowsecurity"),
        text_column("relminmxid"),
        text_column("tminmxid"),
        text_column("reloptions"),
        text_column("checkoption"),
        text_column("amname"),
        bool_column("is_identity_sequence"),
        bool_column("ispartition"),
    ]
}

fn pg_dump_class_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    for table in tables {
        let relhasindex = session
            .indexes
            .iter()
            .any(|index| index.table == table.name && session.tables.contains_key(&index.table));
        rows.push(pg_dump_class_metadata_row(PgDumpClassMetadata {
            session,
            oid: table.oid,
            name: &table.name,
            relkind: "r",
            relchecks: table.check_constraints.len(),
            relhasindex,
            relhasrules: false,
            amname: Some("heap"),
        }));
    }
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by_key(|view| view.oid);
    for view in views {
        rows.push(pg_dump_class_metadata_row(PgDumpClassMetadata {
            session,
            oid: view.oid,
            name: &view.name,
            relkind: "v",
            relchecks: 0,
            relhasindex: false,
            relhasrules: true,
            amname: None,
        }));
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by_key(|view| view.oid);
    for view in materialized_views {
        rows.push(pg_dump_class_metadata_row(PgDumpClassMetadata {
            session,
            oid: view.oid,
            name: &view.name,
            relkind: "m",
            relchecks: 0,
            relhasindex: false,
            relhasrules: true,
            amname: Some("heap"),
        }));
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by_key(|sequence| sequence.oid);
    for sequence in sequences {
        rows.push(pg_dump_class_metadata_row(PgDumpClassMetadata {
            session,
            oid: sequence.oid,
            name: &sequence.name,
            relkind: "S",
            relchecks: 0,
            relhasindex: false,
            relhasrules: false,
            amname: None,
        }));
    }
    rows
}

struct PgDumpClassMetadata<'a> {
    session: &'a Session,
    oid: u32,
    name: &'a str,
    relkind: &'a str,
    relchecks: usize,
    relhasindex: bool,
    relhasrules: bool,
    amname: Option<&'a str>,
}

fn pg_dump_class_metadata_row(metadata: PgDumpClassMetadata<'_>) -> Vec<Option<String>> {
    vec![
        Some("1259".to_string()),
        Some(metadata.oid.to_string()),
        Some(metadata.name.to_string()),
        Some(PUBLIC_NAMESPACE_OID.to_string()),
        Some(metadata.relkind.to_string()),
        Some(metadata.relchecks.to_string()),
        Some("10".to_string()),
        Some("0".to_string()),
        Some(if metadata.relhasindex { "t" } else { "f" }.to_string()),
        Some(if metadata.relhasrules { "t" } else { "f" }.to_string()),
        Some("0".to_string()),
        Some("f".to_string()),
        Some("p".to_string()),
        Some("0".to_string()),
        relation_acl_array_display(metadata.session, metadata.name),
        Some(pg_dump_class_acl_default(metadata.relkind).to_string()),
        Some("0".to_string()),
        Some("0".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some("f".to_string()),
        Some("t".to_string()),
        Some("d".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("0".to_string()),
        None,
        None,
        None,
        metadata.amname.map(str::to_string),
        Some("f".to_string()),
        Some("f".to_string()),
    ]
}

fn pg_dump_class_acl_default(relkind: &str) -> &'static str {
    if relkind == "S" {
        "{postgres=rwU/postgres}"
    } else {
        "{postgres=arwdDxt/postgres}"
    }
}

fn pg_dump_attribute_metadata_query_oids(canonical: &str) -> Option<Vec<u32>> {
    let marker = "from unnest('{";
    let (_, rest) = canonical.split_once(marker)?;
    let (oids, rest) = rest.split_once("}'::pg_catalog.oid[]")?;
    if !rest.contains("join pg_catalog.pg_attribute a") {
        return None;
    }
    if oids.trim().is_empty() {
        return Some(Vec::new());
    }
    oids.split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .ok()
}

fn pg_dump_attribute_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("attrelid"),
        int4_column("attnum"),
        text_column("attname"),
        int4_column("attstattarget"),
        text_column("attstorage"),
        text_column("typstorage"),
        bool_column("attnotnull"),
        bool_column("atthasdef"),
        bool_column("attisdropped"),
        int4_column("attlen"),
        text_column("attalign"),
        bool_column("attislocal"),
        text_column("atttypname"),
        text_column("attoptions"),
        int4_column("attcollation"),
        text_column("attfdwoptions"),
        text_column("attcompression"),
        text_column("attidentity"),
        text_column("attmissingval"),
        text_column("attgenerated"),
    ]
}

fn pg_dump_attribute_metadata_rows(
    session: &Session,
    relation_oids: &[u32],
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for oid in relation_oids {
        if let Some(table) = session.tables.values().find(|table| table.oid == *oid) {
            for column in &table.columns {
                rows.push(pg_dump_attribute_metadata_row(
                    table.oid,
                    column.attnum,
                    &column.def.name,
                    column.def.ty,
                    column.def.domain.as_deref(),
                    column.def.default.is_some(),
                ));
            }
            continue;
        }
        if let Some(view) = session.views.values().find(|view| view.oid == *oid) {
            let Some(table) = session.tables.get(&view.query.table) else {
                continue;
            };
            let columns = match &view.query.projection {
                SelectProjection::All => table.columns.iter().collect::<Vec<_>>(),
                SelectProjection::Columns(names) => names
                    .iter()
                    .filter_map(|name| table.columns.iter().find(|column| column.def.name == *name))
                    .collect::<Vec<_>>(),
                SelectProjection::CountAll
                | SelectProjection::GroupedCount { .. }
                | SelectProjection::Sum { .. }
                | SelectProjection::GroupedSum { .. }
                | SelectProjection::Avg { .. }
                | SelectProjection::GroupedAvg { .. }
                | SelectProjection::Min { .. }
                | SelectProjection::GroupedMin { .. }
                | SelectProjection::Max { .. }
                | SelectProjection::GroupedMax { .. }
                | SelectProjection::CountDistinct { .. }
                | SelectProjection::GroupedAggregates { .. } => Vec::new(),
            };
            for (idx, column) in columns.into_iter().enumerate() {
                let Ok(attnum) = i16::try_from(idx + 1) else {
                    continue;
                };
                rows.push(pg_dump_attribute_metadata_row(
                    view.oid,
                    attnum,
                    &column.def.name,
                    column.def.ty,
                    column.def.domain.as_deref(),
                    false,
                ));
            }
            continue;
        }
        if let Some(view) = session
            .materialized_views
            .values()
            .find(|view| view.oid == *oid)
        {
            for column in &view.columns {
                rows.push(pg_dump_attribute_metadata_row(
                    view.oid,
                    column.attnum,
                    &column.def.name,
                    column.def.ty,
                    column.def.domain.as_deref(),
                    false,
                ));
            }
        }
    }
    rows
}

fn pg_dump_attribute_metadata_row(
    relation_oid: u32,
    attnum: i16,
    name: &str,
    ty: SqlType,
    domain: Option<&str>,
    has_default: bool,
) -> Vec<Option<String>> {
    vec![
        Some(relation_oid.to_string()),
        Some(attnum.to_string()),
        Some(name.to_string()),
        Some("-1".to_string()),
        Some(sql_type_storage_code(ty).to_string()),
        Some(sql_type_storage_code(ty).to_string()),
        Some("f".to_string()),
        Some(if has_default { "t" } else { "f" }.to_string()),
        Some("f".to_string()),
        Some(ty.type_size().to_string()),
        Some(sql_type_alignment_code(ty).to_string()),
        Some("t".to_string()),
        Some(
            domain
                .map(str::to_string)
                .unwrap_or_else(|| sql_type_display_name(ty).to_string()),
        ),
        None,
        Some("0".to_string()),
        None,
        Some(String::new()),
        Some(String::new()),
        None,
        Some(String::new()),
    ]
}

fn is_pg_dump_index_metadata_query(canonical: &str) -> bool {
    canonical.starts_with("select t.tableoid, t.oid, i.indrelid")
        && canonical.contains("join pg_catalog.pg_index i")
}

fn pg_dump_index_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        int4_column("indrelid"),
        text_column("indexname"),
        text_column("indexdef"),
        text_column("indkey"),
        bool_column("indisclustered"),
        text_column("contype"),
        text_column("conname"),
        bool_column("condeferrable"),
        bool_column("condeferred"),
        int4_column("contableoid"),
        int4_column("conoid"),
        text_column("condef"),
        text_column("tablespace"),
        text_column("indreloptions"),
        bool_column("indisreplident"),
        int4_column("parentidx"),
        int4_column("indnkeyatts"),
        int4_column("indnatts"),
        text_column("indstatcols"),
        text_column("indstatvals"),
        bool_column("indnullsnotdistinct"),
    ]
}

fn pg_dump_index_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_index_entries(session)
        .into_iter()
        .map(|entry| {
            vec![
                Some("1259".to_string()),
                Some(entry.index_oid.to_string()),
                Some(entry.table_oid.to_string()),
                Some(entry.index.name.clone()),
                Some(catalog_index_definition(&entry.index)),
                Some(entry.attnum.to_string()),
                Some("f".to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_contype(&entry.index).to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(entry.index.name.clone()),
                Some("f".to_string()),
                Some("f".to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some("2606".to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_oid(&entry).to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_definition(&entry.index)),
                Some(String::new()),
                None,
                Some("f".to_string()),
                Some("0".to_string()),
                Some("1".to_string()),
                Some("1".to_string()),
                None,
                None,
                Some("f".to_string()),
            ]
        })
        .collect()
}

fn is_catalog_foreign_key_metadata_query(canonical: &str) -> bool {
    canonical.starts_with("select c.tableoid, c.oid, conrelid, conname")
        && canonical.contains("join pg_catalog.pg_constraint c")
}

fn catalog_foreign_key_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        int4_column("conrelid"),
        text_column("conname"),
        int4_column("confrelid"),
        int4_column("conindid"),
        text_column("condef"),
    ]
}

fn foreign_key_definition(foreign_key: &CatalogForeignKey) -> String {
    format!(
        "FOREIGN KEY ({}) REFERENCES {}({})",
        foreign_key.column, foreign_key.referenced_table, foreign_key.referenced_column
    )
}

fn catalog_foreign_key_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let index_entries = catalog_index_entries(session);
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    for table in tables {
        for (idx, foreign_key) in table.foreign_keys.iter().enumerate() {
            let referenced_table_oid = session
                .tables
                .get(&foreign_key.referenced_table)
                .map(|table| table.oid)
                .unwrap_or(0);
            let referenced_index_oid = index_entries
                .iter()
                .find(|entry| {
                    entry.index.table == foreign_key.referenced_table
                        && entry.index.column == foreign_key.referenced_column
                        && (entry.index.primary_key || entry.index.unique_constraint)
                })
                .map(|entry| entry.index_oid)
                .unwrap_or(0);
            rows.push(vec![
                Some("2606".to_string()),
                Some(catalog_foreign_key_oid(table.oid, idx).to_string()),
                Some(table.oid.to_string()),
                Some(foreign_key.name.clone()),
                Some(referenced_table_oid.to_string()),
                Some(referenced_index_oid.to_string()),
                Some(foreign_key_definition(foreign_key)),
            ]);
        }
    }
    rows
}

fn pg_dump_view_definition_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select pg_catalog.pg_get_viewdef('";
    let suffix = "'::pg_catalog.oid) as viewdef";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn pg_dump_view_definition_rows(session: &Session, view_oid: u32) -> Vec<Vec<Option<String>>> {
    if let Some(view) = session.views.values().find(|view| view.oid == view_oid) {
        return vec![vec![Some(format!("{};", view.definition))]];
    }
    session
        .materialized_views
        .values()
        .find(|view| view.oid == view_oid)
        .map(|view| vec![vec![Some(format!("{};", view.definition))]])
        .unwrap_or_default()
}

fn pg_dump_dependency_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by_key(|view| view.oid);
    for view in views {
        if let Some(table) = session.tables.get(&view.query.table) {
            rows.push(vec![
                Some("1259".to_string()),
                Some(view.oid.to_string()),
                Some("1259".to_string()),
                Some(table.oid.to_string()),
                Some("n".to_string()),
            ]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by_key(|view| view.oid);
    for view in materialized_views {
        if let Some(table) = session.tables.get(&view.query.table) {
            rows.push(vec![
                Some("1259".to_string()),
                Some(view.oid.to_string()),
                Some("1259".to_string()),
                Some(table.oid.to_string()),
                Some("n".to_string()),
            ]);
        }
    }
    rows
}

#[derive(Clone)]
struct CatalogIndexEntry {
    table_oid: u32,
    attnum: i16,
    index_oid: u32,
    index: CatalogIndex,
}

fn catalog_index_entries(session: &Session) -> Vec<CatalogIndexEntry> {
    let mut indexed = session
        .indexes
        .iter()
        .filter_map(|index| {
            let table = session.tables.get(&index.table)?;
            let column = table
                .columns
                .iter()
                .find(|column| column.def.name == index.column)?;
            Some((table.oid, table.name.clone(), column.attnum, index.clone()))
        })
        .collect::<Vec<_>>();
    indexed.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.3.name.cmp(&right.3.name))
    });
    indexed
        .into_iter()
        .enumerate()
        .map(
            |(idx, (table_oid, _table_name, attnum, index))| CatalogIndexEntry {
                table_oid,
                attnum,
                index_oid: FIRST_USER_INDEX_OID + idx as u32,
                index,
            },
        )
        .collect()
}

fn catalog_constraint_entries(session: &Session) -> Vec<CatalogIndexEntry> {
    catalog_index_entries(session)
        .into_iter()
        .filter(|entry| entry.index.primary_key || entry.index.unique_constraint)
        .collect()
}

fn catalog_constraint_type(index: &CatalogIndex) -> &'static str {
    if index.primary_key {
        "PRIMARY KEY"
    } else {
        "UNIQUE"
    }
}

fn catalog_constraint_contype(index: &CatalogIndex) -> &'static str {
    if index.primary_key {
        "p"
    } else {
        "u"
    }
}

fn catalog_constraint_definition(index: &CatalogIndex) -> String {
    format!("{} ({})", catalog_constraint_type(index), index.column)
}

fn catalog_constraint_oid(entry: &CatalogIndexEntry) -> u32 {
    40_000 + entry.index_oid
}

fn catalog_foreign_key_oid(table_oid: u32, idx: usize) -> u32 {
    80_000 + table_oid + idx as u32
}

fn catalog_index_oid(session: &Session, index_name: &str) -> Option<u32> {
    catalog_index_entries(session)
        .into_iter()
        .find(|entry| entry.index.name == index_name)
        .map(|entry| entry.index_oid)
}

fn pg_dump_attrdef_metadata_query_relation_oids(canonical: &str) -> Option<Vec<u32>> {
    let prefix = "select a.tableoid, a.oid, adrelid, adnum, pg_catalog.pg_get_expr(adbin, adrelid) as adsrc from unnest('{";
    let suffix = "}'::pg_catalog.oid[]) as src(tbloid) join pg_catalog.pg_attrdef a on (src.tbloid = a.adrelid) order by a.adrelid, a.adnum";
    let body = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let oids = body
        .split(',')
        .map(|raw| raw.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (!oids.is_empty()).then_some(oids)
}

fn pg_dump_attrdef_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        int4_column("adrelid"),
        int4_column("adnum"),
        text_column("adsrc"),
    ]
}

fn pg_dump_attrdef_metadata_rows(
    session: &Session,
    relation_oids: &[u32],
) -> Vec<Vec<Option<String>>> {
    let requested = relation_oids.iter().copied().collect::<BTreeSet<_>>();
    let mut tables = session
        .tables
        .values()
        .filter(|table| requested.contains(&table.oid))
        .collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().filter_map(|column| {
                column.def.default.as_ref().map(|default| {
                    vec![
                        Some("2604".to_string()),
                        Some((30_000_u32 + table.oid + column.attnum as u32).to_string()),
                        Some(table.oid.to_string()),
                        Some(column.attnum.to_string()),
                        Some(format_column_default_expr(default)),
                    ]
                })
            })
        })
        .collect()
}

fn pg_dump_sequence_metadata_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select format_type(seqtypid, null), seqstart, seqincrement, seqmax, seqmin, seqcache, seqcycle from pg_catalog.pg_sequence where seqrelid = '";
    let suffix = "'::oid";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn pg_dump_sequence_metadata_columns() -> Vec<Column> {
    vec![
        text_column("format_type"),
        int8_column("seqstart"),
        int8_column("seqincrement"),
        int8_column("seqmax"),
        int8_column("seqmin"),
        int8_column("seqcache"),
        bool_column("seqcycle"),
    ]
}

fn pg_dump_sequence_metadata_rows(
    session: &Session,
    sequence_oid: u32,
) -> Vec<Vec<Option<String>>> {
    if session
        .sequences
        .values()
        .any(|sequence| sequence.oid == sequence_oid)
    {
        return vec![vec![
            Some("bigint".to_string()),
            Some("1".to_string()),
            Some("1".to_string()),
            Some("9223372036854775807".to_string()),
            Some("1".to_string()),
            Some("1".to_string()),
            Some("f".to_string()),
        ]];
    }
    Vec::new()
}

fn pg_dump_sequence_last_value_query_name(canonical: &str) -> Option<String> {
    let prefix = "select last_value, is_called from ";
    let name = canonical.strip_prefix(prefix)?.trim();
    if name.contains(' ') || name.contains('(') || name.contains(')') {
        return None;
    }
    Some(name.strip_prefix("public.").unwrap_or(name).to_string())
}

fn pg_dump_sequence_setval_query(canonical: &str) -> Option<(String, i64, bool)> {
    let prefix = "select pg_catalog.setval('";
    let rest = canonical.strip_prefix(prefix)?;
    let (name, rest) = rest.split_once("',")?;
    let (value, is_called) = rest.strip_suffix(')')?.split_once(',')?;
    let value = value.trim().parse::<i64>().ok()?;
    let is_called = if is_called.trim().eq_ignore_ascii_case("true")
        || is_called.trim().eq_ignore_ascii_case("t")
    {
        true
    } else if is_called.trim().eq_ignore_ascii_case("false")
        || is_called.trim().eq_ignore_ascii_case("f")
    {
        false
    } else {
        return None;
    };
    Some((
        name.strip_prefix("public.").unwrap_or(name).to_string(),
        value,
        is_called,
    ))
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

fn sql_type_alignment_code(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Int2 => "s",
        SqlType::Int4 => "i",
        SqlType::Int8 => "d",
        SqlType::Numeric { .. } => "i",
        SqlType::Bool => "c",
        SqlType::Text => "i",
        SqlType::Date => "i",
        SqlType::Timestamp => "d",
        SqlType::Uuid => "c",
    }
}

fn is_pg_dump_function_metadata_query(canonical: &str) -> bool {
    canonical
        .starts_with("select p.tableoid, p.oid, p.proname, p.prolang, p.pronargs, p.proargtypes")
        && canonical.contains("from pg_proc p")
}

fn is_pg_dump_function_dump_prepare(canonical: &str) -> bool {
    canonical.starts_with("prepare dumpfunc(pg_catalog.oid) as select proretset, prosrc, probin")
        && canonical.contains("from pg_catalog.pg_proc p, pg_catalog.pg_language l")
}

fn pg_dump_function_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("proname"),
        int4_column("prolang"),
        int4_column("pronargs"),
        text_column("proargtypes"),
        int4_column("prorettype"),
        text_column("proacl"),
        text_column("acldefault"),
        int4_column("pronamespace"),
        int4_column("proowner"),
    ]
}

fn pg_dump_function_dump_columns() -> Vec<Column> {
    vec![
        bool_column("proretset"),
        text_column("prosrc"),
        text_column("probin"),
        text_column("provolatile"),
        bool_column("proisstrict"),
        bool_column("prosecdef"),
        text_column("lanname"),
        text_column("proconfig"),
        text_column("procost"),
        text_column("prorows"),
        text_column("funcargs"),
        text_column("funciargs"),
        text_column("funcresult"),
        bool_column("proleakproof"),
        text_column("protrftypes"),
        text_column("proparallel"),
        text_column("prokind"),
        text_column("prosupport"),
        text_column("prosqlbody"),
    ]
}

fn pg_dump_function_dump_rows(session: &Session, oid: Option<u32>) -> Vec<Vec<Option<String>>> {
    session
        .functions
        .values()
        .find(|function| Some(function.oid) == oid)
        .map(|function| {
            vec![vec![
                Some("f".to_string()),
                Some(function.body.clone()),
                None,
                Some("v".to_string()),
                Some("f".to_string()),
                Some("f".to_string()),
                Some("sql".to_string()),
                None,
                Some("100".to_string()),
                Some("0".to_string()),
                Some(String::new()),
                Some(String::new()),
                Some(sql_type_display_name(function.return_type).to_string()),
                Some("f".to_string()),
                None,
                Some("u".to_string()),
                Some("f".to_string()),
                Some("-".to_string()),
                None,
            ]]
        })
        .unwrap_or_default()
}

fn pg_dump_function_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .functions
        .values()
        .map(|function| {
            vec![
                Some("1255".to_string()),
                Some(function.oid.to_string()),
                Some(function.name.clone()),
                Some("14".to_string()),
                Some("0".to_string()),
                Some(String::new()),
                Some(function.return_type.postgres_oid().to_string()),
                function_acl_array_display(&function.acl),
                Some("{=X/postgres,postgres=X/postgres}".to_string()),
                Some(PUBLIC_NAMESPACE_OID.to_string()),
                Some("10".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[2].cmp(&right[2]));
    rows
}

fn pg_dump_empty_catalog_query_columns(canonical: &str) -> Option<Vec<Column>> {
    if canonical == "select distinct attrelid from pg_attribute where attacl is not null" {
        return Some(vec![int4_column("attrelid")]);
    }
    if canonical == "select objoid, classoid, objsubid, privtype, initprivs from pg_init_privs" {
        return Some(vec![
            int4_column("objoid"),
            int4_column("classoid"),
            int4_column("objsubid"),
            text_column("privtype"),
            text_column("initprivs"),
        ]);
    }
    if is_pg_dump_function_metadata_query(canonical) {
        return Some(pg_dump_function_metadata_columns());
    }
    if canonical.starts_with("select p.tableoid, p.oid, p.proname as aggname")
        && canonical.contains("from pg_proc p")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("aggname"),
            int4_column("aggnamespace"),
            int4_column("pronargs"),
            text_column("proargtypes"),
            int4_column("proowner"),
            text_column("aggacl"),
            text_column("acldefault"),
        ]);
    }
    if canonical
        == "select tableoid, oid, lanname, lanpltrusted, lanplcallfoid, laninline, lanvalidator, lanacl, acldefault('l', lanowner) as acldefault, lanowner from pg_language where lanispl order by oid"
    {
        return Some(pg_language_discovery_columns());
    }
    if canonical.starts_with("select provider, label from pg_catalog.pg_shseclabel where ") {
        return Some(vec![text_column("provider"), text_column("label")]);
    }
    if canonical.starts_with("select unnest(setconfig) from pg_db_role_setting ") {
        return Some(vec![text_column("unnest")]);
    }
    if canonical.starts_with("select rolname, unnest(setconfig) from pg_db_role_setting ") {
        return Some(vec![text_column("rolname"), text_column("unnest")]);
    }
    if canonical.starts_with("select ur.rolname as role, um.rolname as member")
        && canonical.contains("from pg_auth_members a")
    {
        return Some(vec![
            text_column("role"),
            text_column("member"),
            text_column("grantor"),
            int4_column("roleid"),
            int4_column("memberid"),
            int4_column("grantorid"),
            bool_column("admin_option"),
            bool_column("inherit_option"),
            bool_column("set_option"),
        ]);
    }
    if canonical.starts_with("select parname, pg_catalog.pg_get_userbyid(10) as parowner")
        && canonical.contains("from pg_catalog.pg_parameter_acl")
    {
        return Some(vec![
            text_column("parname"),
            text_column("parowner"),
            text_column("paracl"),
            text_column("acldefault"),
        ]);
    }
    if is_pg_dumpall_tablespace_metadata_query(canonical) {
        return Some(pg_dumpall_tablespace_metadata_columns());
    }
    if canonical == "select tableoid, oid, oprname, oprnamespace, oprowner, oprkind, oprleft, oprright, oprcode::oid as oprcode from pg_operator" {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("oprname"),
            int4_column("oprnamespace"),
            int4_column("oprowner"),
            text_column("oprkind"),
            int4_column("oprleft"),
            int4_column("oprright"),
            int4_column("oprcode"),
        ]);
    }
    if canonical
        == "select tableoid, oid, amname, amtype, amhandler::pg_catalog.regproc as amhandler from pg_am"
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("amname"),
            text_column("amtype"),
            text_column("amhandler"),
        ]);
    }
    if canonical
        == "select tableoid, oid, opcmethod, opcname, opcnamespace, opcowner from pg_opclass"
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            int4_column("opcmethod"),
            text_column("opcname"),
            int4_column("opcnamespace"),
            int4_column("opcowner"),
        ]);
    }
    if canonical
        == "select tableoid, oid, opfmethod, opfname, opfnamespace, opfowner from pg_opfamily"
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            int4_column("opfmethod"),
            text_column("opfname"),
            int4_column("opfnamespace"),
            int4_column("opfowner"),
        ]);
    }
    if canonical == "select tableoid, oid, prsname, prsnamespace, prsstart::oid, prstoken::oid, prsend::oid, prsheadline::oid, prslextype::oid from pg_ts_parser" {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("prsname"),
            int4_column("prsnamespace"),
            int4_column("prsstart"),
            int4_column("prstoken"),
            int4_column("prsend"),
            int4_column("prsheadline"),
            int4_column("prslextype"),
        ]);
    }
    if canonical
        == "select tableoid, oid, tmplname, tmplnamespace, tmplinit::oid, tmpllexize::oid from pg_ts_template"
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("tmplname"),
            int4_column("tmplnamespace"),
            int4_column("tmplinit"),
            int4_column("tmpllexize"),
        ]);
    }
    if canonical
        == "select tableoid, oid, dictname, dictnamespace, dictowner, dicttemplate, dictinitoption from pg_ts_dict"
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("dictname"),
            int4_column("dictnamespace"),
            int4_column("dictowner"),
            int4_column("dicttemplate"),
            text_column("dictinitoption"),
        ]);
    }
    if canonical
        == "select tableoid, oid, cfgname, cfgnamespace, cfgowner, cfgparser from pg_ts_config"
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("cfgname"),
            int4_column("cfgnamespace"),
            int4_column("cfgowner"),
            int4_column("cfgparser"),
        ]);
    }
    if canonical.starts_with("select tableoid, oid, fdwname, fdwowner")
        && canonical.contains("from pg_foreign_data_wrapper")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("fdwname"),
            int4_column("fdwowner"),
            text_column("fdwhandler"),
            text_column("fdwvalidator"),
            text_column("fdwacl"),
            text_column("acldefault"),
            text_column("fdwoptions"),
        ]);
    }
    if canonical.starts_with("select tableoid, oid, srvname, srvowner")
        && canonical.contains("from pg_foreign_server")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("srvname"),
            int4_column("srvowner"),
            int4_column("srvfdw"),
            text_column("srvtype"),
            text_column("srvversion"),
            text_column("srvacl"),
            text_column("acldefault"),
            text_column("srvoptions"),
        ]);
    }
    if canonical == "select tableoid, oid, collname, collnamespace, collowner, collencoding from pg_collation" {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("collname"),
            int4_column("collnamespace"),
            int4_column("collowner"),
            int4_column("collencoding"),
        ]);
    }
    if canonical == "select tableoid, oid, conname, connamespace, conowner from pg_conversion" {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("conname"),
            int4_column("connamespace"),
            int4_column("conowner"),
        ]);
    }
    if canonical.starts_with("select tableoid, oid, castsource, casttarget")
        && canonical.contains("from pg_cast")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            int4_column("castsource"),
            int4_column("casttarget"),
            int4_column("castfunc"),
            text_column("castcontext"),
            text_column("castmethod"),
        ]);
    }
    if canonical == "select tableoid, oid, trftype, trflang, trffromsql::oid, trftosql::oid from pg_transform order by 3,4" {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            int4_column("trftype"),
            int4_column("trflang"),
            int4_column("trffromsql"),
            int4_column("trftosql"),
        ]);
    }
    if canonical == "select inhrelid, inhparent from pg_inherits" {
        return Some(vec![int4_column("inhrelid"), int4_column("inhparent")]);
    }
    if canonical.starts_with("select partrelid from pg_partitioned_table") {
        return Some(vec![int4_column("partrelid")]);
    }
    if canonical.starts_with("select t.tableoid, t.oid, i.indrelid")
        && canonical.contains("join pg_catalog.pg_index i")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            int4_column("indrelid"),
            text_column("indexname"),
            text_column("indexdef"),
            text_column("indkey"),
            bool_column("indisclustered"),
            text_column("contype"),
            text_column("conname"),
            bool_column("condeferrable"),
            bool_column("condeferred"),
            int4_column("contableoid"),
            int4_column("conoid"),
            text_column("condef"),
            text_column("tablespace"),
            text_column("indreloptions"),
            bool_column("indisreplident"),
            int4_column("parentidx"),
            int4_column("indnkeyatts"),
            int4_column("indnatts"),
            text_column("indstatcols"),
            text_column("indstatvals"),
            bool_column("indnullsnotdistinct"),
        ]);
    }
    if canonical == "select tableoid, oid, stxname, stxnamespace, stxowner, stxrelid, stxstattarget from pg_catalog.pg_statistic_ext" {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("stxname"),
            int4_column("stxnamespace"),
            int4_column("stxowner"),
            int4_column("stxrelid"),
            int4_column("stxstattarget"),
        ]);
    }
    if canonical.starts_with("select c.tableoid, c.oid, conrelid, conname")
        && canonical.contains("join pg_catalog.pg_constraint c")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            int4_column("conrelid"),
            text_column("conname"),
            int4_column("confrelid"),
            int4_column("conindid"),
            text_column("condef"),
        ]);
    }
    if canonical.starts_with("select t.tgrelid, t.tgname")
        && canonical.contains("join pg_catalog.pg_trigger t")
    {
        return Some(vec![
            int4_column("tgrelid"),
            text_column("tgname"),
            text_column("tgfname"),
            text_column("tgdef"),
            text_column("tgenabled"),
            int4_column("tableoid"),
            int4_column("oid"),
            bool_column("tgispartition"),
        ]);
    }
    if canonical == "select tableoid, oid, rulename, ev_class as ruletable, ev_type, is_instead, ev_enabled from pg_rewrite order by oid" {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("rulename"),
            int4_column("ruletable"),
            text_column("ev_type"),
            bool_column("is_instead"),
            text_column("ev_enabled"),
        ]);
    }
    if canonical.starts_with("select pol.oid, pol.tableoid, pol.polrelid")
        && canonical.contains("join pg_catalog.pg_policy pol")
    {
        return Some(vec![
            int4_column("oid"),
            int4_column("tableoid"),
            int4_column("polrelid"),
            text_column("polname"),
            text_column("polcmd"),
            bool_column("polpermissive"),
            text_column("polroles"),
            text_column("polqual"),
            text_column("polwithcheck"),
        ]);
    }
    if canonical.starts_with("select p.tableoid, p.oid, p.pubname")
        && canonical.contains("from pg_publication p")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("pubname"),
            int4_column("pubowner"),
            bool_column("puballtables"),
            bool_column("pubinsert"),
            bool_column("pubupdate"),
            bool_column("pubdelete"),
            bool_column("pubtruncate"),
            bool_column("pubviaroot"),
        ]);
    }
    if canonical.starts_with("select tableoid, oid, prpubid, prrelid")
        && canonical.contains("from pg_catalog.pg_publication_rel pr")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            int4_column("prpubid"),
            int4_column("prrelid"),
            text_column("prrelqual"),
            text_column("prattrs"),
        ]);
    }
    if canonical
        == "select tableoid, oid, pnpubid, pnnspid from pg_catalog.pg_publication_namespace"
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            int4_column("pnpubid"),
            int4_column("pnnspid"),
        ]);
    }
    if canonical.starts_with("select e.tableoid, e.oid, evtname")
        && canonical.contains("from pg_event_trigger e")
    {
        return Some(vec![
            int4_column("tableoid"),
            int4_column("oid"),
            text_column("evtname"),
            text_column("evtenabled"),
            text_column("evtevent"),
            int4_column("evtowner"),
            text_column("evttags"),
            text_column("evtfname"),
        ]);
    }
    None
}

fn is_pg_dump_domain_constraints_prepare(canonical: &str) -> bool {
    canonical
        == "prepare getdomainconstraints(pg_catalog.oid) as select tableoid, oid, conname, pg_catalog.pg_get_constraintdef(oid) as consrc, convalidated from pg_catalog.pg_constraint where contypid = $1 order by conname"
}

fn is_pg_dump_domain_constraints_execute(canonical: &str) -> bool {
    canonical.starts_with("execute getdomainconstraints(") && canonical.ends_with(')')
}

fn pg_dump_domain_constraints_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("conname"),
        text_column("consrc"),
        bool_column("convalidated"),
    ]
}

fn is_pg_dump_domain_dump_prepare(canonical: &str) -> bool {
    canonical
        == "prepare dumpdomain(pg_catalog.oid) as select t.typnotnull, pg_catalog.format_type(t.typbasetype, t.typtypmod) as typdefn, pg_catalog.pg_get_expr(t.typdefaultbin, 'pg_catalog.pg_type'::pg_catalog.regclass) as typdefaultbin, t.typdefault, case when t.typcollation <> u.typcollation then t.typcollation else 0 end as typcollation from pg_catalog.pg_type t left join pg_catalog.pg_type u on (t.typbasetype = u.oid) where t.oid = $1"
}

fn pg_dump_domain_dump_execute_oid(canonical: &str) -> Option<u32> {
    canonical
        .strip_prefix("execute dumpdomain(")?
        .strip_suffix(')')?
        .trim_matches('\'')
        .parse()
        .ok()
}

fn pg_dump_domain_dump_columns() -> Vec<Column> {
    vec![
        bool_column("typnotnull"),
        text_column("typdefn"),
        text_column("typdefaultbin"),
        text_column("typdefault"),
        int4_column("typcollation"),
    ]
}

fn pg_dump_domain_dump_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
    let Some(domain) = session.domains.values().find(|domain| domain.oid == oid) else {
        return Vec::new();
    };
    vec![vec![
        Some("f".to_string()),
        Some(sql_type_display_name(domain.base_type).to_string()),
        None,
        None,
        Some("0".to_string()),
    ]]
}

fn pg_dump_type_metadata_query() -> &'static str {
    "select tableoid, oid, typname, typnamespace, typacl, acldefault('t', typowner) as acldefault, typowner, typelem, typrelid, case when typrelid = 0 then ' '::\"char\" else (select relkind from pg_class where oid = typrelid) end as typrelkind, typtype, typisdefined, typname[0] = '_' and typelem != 0 and (select typarray from pg_type te where oid = pg_type.typelem) = oid as isarray from pg_type"
}

fn pg_dump_type_metadata_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("typname"),
        int4_column("typnamespace"),
        text_column("typacl"),
        text_column("acldefault"),
        int4_column("typowner"),
        int4_column("typelem"),
        int4_column("typrelid"),
        text_column("typrelkind"),
        text_column("typtype"),
        bool_column("typisdefined"),
        bool_column("isarray"),
    ]
}

fn pg_dump_type_metadata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = SUPPORTED_SQL_TYPES
        .into_iter()
        .map(|ty| {
            vec![
                Some("1247".to_string()),
                Some(ty.postgres_oid().to_string()),
                Some(ty.catalog_name().to_string()),
                Some("11".to_string()),
                None,
                None,
                Some("10".to_string()),
                Some("0".to_string()),
                Some("0".to_string()),
                Some(" ".to_string()),
                Some("b".to_string()),
                Some("t".to_string()),
                Some("f".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    let mut domains = session.domains.values().collect::<Vec<_>>();
    domains.sort_by_key(|domain| domain.oid);
    for domain in domains {
        rows.push(vec![
            Some("1247".to_string()),
            Some(domain.oid.to_string()),
            Some(domain.name.clone()),
            Some(PUBLIC_NAMESPACE_OID.to_string()),
            None,
            None,
            Some("10".to_string()),
            Some("0".to_string()),
            Some("0".to_string()),
            Some(" ".to_string()),
            Some("d".to_string()),
            Some("t".to_string()),
            Some("f".to_string()),
        ]);
    }
    rows
}

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

fn catalog_describe_relation_lookup_rows(
    session: &Session,
    relname_pattern: &str,
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    for table in tables
        .into_iter()
        .filter(|table| psql_relname_pattern_matches(relname_pattern, &table.name))
    {
        rows.push(vec![
            Some(table.oid.to_string()),
            Some("public".to_string()),
            Some(table.name.clone()),
        ]);
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    for sequence in sequences
        .into_iter()
        .filter(|sequence| psql_relname_pattern_matches(relname_pattern, &sequence.name))
    {
        rows.push(vec![
            Some(sequence.oid.to_string()),
            Some("public".to_string()),
            Some(sequence.name.clone()),
        ]);
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in materialized_views
        .into_iter()
        .filter(|view| psql_relname_pattern_matches(relname_pattern, &view.name))
    {
        rows.push(vec![
            Some(view.oid.to_string()),
            Some("public".to_string()),
            Some(view.name.clone()),
        ]);
    }
    rows
}

fn catalog_describe_relation_lookup_rows_for_public_namespace(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    for table in tables {
        rows.push(vec![
            Some(table.oid.to_string()),
            Some("public".to_string()),
            Some(table.name.clone()),
        ]);
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    for sequence in sequences {
        rows.push(vec![
            Some(sequence.oid.to_string()),
            Some("public".to_string()),
            Some(sequence.name.clone()),
        ]);
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in materialized_views {
        rows.push(vec![
            Some(view.oid.to_string()),
            Some("public".to_string()),
            Some(view.name.clone()),
        ]);
    }
    rows
}

fn catalog_describe_relation_flags_query_oid(canonical: &str) -> Option<u32> {
    let plain_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, '', c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let verbose_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, pg_catalog.array_to_string(c.reloptions || array(select 'toast.' || x from pg_catalog.unnest(tc.reloptions) x), ', '), c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let verbose_wrapped_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, pg_catalog.array_to_string(c.reloptions || array(select 'toast.' || x from pg_catalog.unnest(tc.reloptions) x), ', ') , c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let suffix = "'";
    canonical
        .strip_prefix(plain_prefix)
        .or_else(|| canonical.strip_prefix(verbose_prefix))
        .or_else(|| canonical.strip_prefix(verbose_wrapped_prefix))?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_relation_flags_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
    if session
        .sequences
        .values()
        .any(|sequence| sequence.oid == oid)
    {
        return vec![vec![
            Some("0".to_string()),
            Some("s".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some(String::new()),
            Some("0".to_string()),
            Some(String::new()),
            Some("p".to_string()),
            Some("d".to_string()),
            None,
        ]];
    }
    if session
        .materialized_views
        .values()
        .any(|view| view.oid == oid)
    {
        return vec![vec![
            Some("0".to_string()),
            Some("m".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some("f".to_string()),
            Some(String::new()),
            Some("0".to_string()),
            Some(String::new()),
            Some("p".to_string()),
            Some("d".to_string()),
            Some("heap".to_string()),
        ]];
    }
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    let relhasindex = session
        .indexes
        .iter()
        .any(|index| index.table == table.name);
    let relhastriggers = !table.foreign_keys.is_empty()
        || session.tables.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        });

    vec![vec![
        Some(table.check_constraints.len().to_string()),
        Some("r".to_string()),
        Some(if relhasindex { "t" } else { "f" }.to_string()),
        Some("f".to_string()),
        Some(if relhastriggers { "t" } else { "f" }.to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some(String::new()),
        Some("0".to_string()),
        Some(String::new()),
        Some("p".to_string()),
        Some("d".to_string()),
        Some("heap".to_string()),
    ]]
}

fn catalog_describe_attribute_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated from pg_catalog.pg_attribute a where a.attrelid = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_index_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select c2.relname, i.indisprimary, i.indisunique, i.indisclustered, i.indisvalid, pg_catalog.pg_get_indexdef(i.indexrelid, 0, true), pg_catalog.pg_get_constraintdef(con.oid, true), contype, condeferrable, condeferred, i.indisreplident, c2.reltablespace from pg_catalog.pg_class c, pg_catalog.pg_class c2, pg_catalog.pg_index i left join pg_catalog.pg_constraint con on (conrelid = i.indrelid and conindid = i.indexrelid and contype in ('p','u','x')) where c.oid = '";
    let suffix = "' and c.oid = i.indrelid and i.indexrelid = c2.oid order by i.indisprimary desc, c2.relname";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_check_constraints_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select r.conname, pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where r.conrelid = '";
    let suffix = "' and r.contype = 'c' order by 1";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_foreign_keys_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_constraint r")
        || !canonical.contains("r.contype = 'f'")
        || !canonical.contains("conparentid = 0")
    {
        return None;
    }
    let marker = "r.conrelid = '";
    let (_, rest) = canonical.split_once(marker)?;
    let (oid, _) = rest.split_once('\'')?;
    oid.parse().ok()
}

fn catalog_describe_referenced_by_foreign_keys_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_constraint c")
        || !canonical.contains("confrelid in (select pg_catalog.pg_partition_ancestors('")
        || !canonical.contains("and contype = 'f'")
        || !canonical.contains("conparentid = 0")
    {
        return None;
    }
    let marker = "pg_partition_ancestors('";
    let (_, rest) = canonical.split_once(marker)?;
    let (oid, _) = rest.split_once('\'')?;
    oid.parse().ok()
}

fn catalog_describe_trigger_query_oid(canonical: &str) -> Option<u32> {
    if !canonical.contains("from pg_catalog.pg_trigger t")
        || !canonical.contains("pg_catalog.pg_get_triggerdef(t.oid, true)")
    {
        return None;
    }
    let marker = "where t.tgrelid = '";
    let (_, rest) = canonical.split_once(marker)?;
    let (oid, _) = rest.split_once('\'')?;
    oid.parse().ok()
}

fn check_constraint_definition(constraint: &CatalogCheckConstraint) -> String {
    let op = match constraint.op {
        SelectFilterOp::Eq => "=",
        SelectFilterOp::Lt => "<",
        SelectFilterOp::Lte => "<=",
        SelectFilterOp::Gt => ">",
        SelectFilterOp::Gte => ">=",
        SelectFilterOp::LikePrefix => "LIKE",
    };
    let value = match &constraint.value {
        SqlValue::Text(value) => format!("'{}'", value.replace('\'', "''")),
        value => format_sql_value(value),
    };
    format!("CHECK (({} {} {}))", constraint.column, op, value)
}

fn catalog_describe_foreign_key_rows(
    session: &Session,
    table_oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == table_oid) else {
        return Vec::new();
    };
    let mut rows = table
        .foreign_keys
        .iter()
        .map(|constraint| {
            vec![
                Some("t".to_string()),
                Some(constraint.name.clone()),
                Some(foreign_key_definition(constraint)),
                Some(format!("public.{}", table.name)),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn catalog_describe_referenced_by_foreign_key_rows(
    session: &Session,
    table_oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(parent_table) = session.tables.values().find(|table| table.oid == table_oid) else {
        return Vec::new();
    };
    let mut rows = session
        .tables
        .values()
        .flat_map(|table| {
            table
                .foreign_keys
                .iter()
                .filter(|constraint| constraint.referenced_table == parent_table.name)
                .map(|constraint| {
                    vec![
                        Some(constraint.name.clone()),
                        Some(format!("public.{}", table.name)),
                        Some(foreign_key_definition(constraint)),
                    ]
                })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[0].cmp(&right[0]).then_with(|| left[1].cmp(&right[1])));
    rows
}

fn catalog_describe_check_constraint_rows(
    session: &Session,
    table_oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == table_oid) else {
        return Vec::new();
    };
    let mut rows = table
        .check_constraints
        .iter()
        .map(|constraint| {
            vec![
                Some(constraint.name.clone()),
                Some(check_constraint_definition(constraint)),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[0].cmp(&right[0]));
    rows
}

fn catalog_describe_index_rows(session: &Session, table_oid: u32) -> Vec<Vec<Option<String>>> {
    catalog_index_entries(session)
        .into_iter()
        .filter(|entry| entry.table_oid == table_oid)
        .map(|entry| {
            vec![
                Some(entry.index.name.clone()),
                Some(if entry.index.primary_key { "t" } else { "f" }.to_string()),
                Some(if entry.index.unique { "t" } else { "f" }.to_string()),
                Some("f".to_string()),
                Some("t".to_string()),
                Some(catalog_index_definition(&entry.index)),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_definition(&entry.index)),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some(catalog_constraint_contype(&entry.index).to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some("f".to_string()),
                (entry.index.primary_key || entry.index.unique_constraint)
                    .then_some("f".to_string()),
                Some("f".to_string()),
                Some("0".to_string()),
            ]
        })
        .collect()
}

fn catalog_describe_verbose_attribute_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated, a.attstorage, a.attcompression as attcompression, case when a.attstattarget=-1 then null else a.attstattarget end as attstattarget, pg_catalog.col_description(a.attrelid, a.attnum) from pg_catalog.pg_attribute a where a.attrelid = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_verbose_attribute_rows(
    session: &Session,
    oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(column_type_display_name(column)),
                column.def.default.as_ref().map(format_column_default_expr),
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
                Some(sql_type_storage_code(column.def.ty).to_string()),
                Some(String::new()),
                None,
                session
                    .comments
                    .get(&CatalogCommentTarget::Column {
                        table: table.name.clone(),
                        attnum: column.attnum,
                    })
                    .cloned(),
            ]
        })
        .collect()
}

fn catalog_describe_attribute_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(column_type_display_name(column)),
                column.def.default.as_ref().map(format_column_default_expr),
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
            ]
        })
        .collect()
}

fn sql_type_storage_code(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Int2 => "p",
        SqlType::Int4 => "p",
        SqlType::Int8 => "p",
        SqlType::Numeric { .. } => "m",
        SqlType::Bool => "p",
        SqlType::Text => "x",
        SqlType::Date => "p",
        SqlType::Timestamp => "p",
        SqlType::Uuid => "p",
    }
}

fn sql_type_display_name(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Int2 => "smallint",
        SqlType::Int4 => "integer",
        SqlType::Int8 => "bigint",
        SqlType::Numeric { .. } => "numeric",
        SqlType::Bool => "boolean",
        SqlType::Text => "text",
        SqlType::Date => "date",
        SqlType::Timestamp => "timestamp without time zone",
        SqlType::Uuid => "uuid",
    }
}

fn column_type_display_name(column: &CatalogColumn) -> String {
    column
        .def
        .domain
        .clone()
        .unwrap_or_else(|| sql_type_display_name(column.def.ty).to_string())
}

fn column_type_oid(session: &Session, column: &CatalogColumn) -> u32 {
    column
        .def
        .domain
        .as_ref()
        .and_then(|domain| session.domains.get(domain))
        .map(|domain| domain.oid)
        .unwrap_or_else(|| column.def.ty.postgres_oid())
}

fn column_type_size(column: &CatalogColumn) -> i16 {
    column.def.ty.type_size()
}

fn information_schema_udt_metadata(column: &CatalogColumn) -> (String, String) {
    if let Some(domain) = column.def.domain.as_ref() {
        ("public".to_string(), domain.clone())
    } else {
        (
            "pg_catalog".to_string(),
            column.def.ty.catalog_name().to_string(),
        )
    }
}

fn sql_type_by_oid(oid: u32) -> Option<SqlType> {
    SUPPORTED_SQL_TYPES
        .into_iter()
        .find(|ty| ty.postgres_oid() == oid)
}

fn psql_describe_query_type_rows(canonical: &str) -> Option<Vec<Vec<Option<String>>>> {
    let prefix =
        "select name as \"column\", pg_catalog.format_type(tp, tpm) as \"type\" from (values ";
    let suffix = ") s(name, tp, tpm)";
    let values = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut rows = Vec::new();
    for raw_value in values.split("),(") {
        let value = raw_value
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')');
        let fields = value.split(',').map(str::trim).collect::<Vec<_>>();
        let [name, oid, _typmod] = fields.as_slice() else {
            return None;
        };
        let name = name.strip_prefix('\'')?.strip_suffix('\'')?;
        let oid = oid.strip_prefix('\'')?.strip_suffix("'::pg_catalog.oid")?;
        let oid = oid.parse::<u32>().ok()?;
        let ty = sql_type_by_oid(oid)?;
        rows.push(vec![
            Some(name.to_string()),
            Some(sql_type_display_name(ty).to_string()),
        ]);
    }
    Some(rows)
}

fn catalog_describe_policy_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select pol.polname, pol.polpermissive, case when pol.polroles = '{0}' then null else pg_catalog.array_to_string(array(select rolname from pg_catalog.pg_roles where oid = any (pol.polroles) order by 1),',') end, pg_catalog.pg_get_expr(pol.polqual, pol.polrelid), pg_catalog.pg_get_expr(pol.polwithcheck, pol.polrelid), case pol.polcmd when 'r' then 'select' when 'a' then 'insert' when 'w' then 'update' when 'd' then 'delete' end as cmd from pg_catalog.pg_policy pol where pol.polrelid = '";
    let suffix = "' order by 1";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_statistic_ext_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select oid, stxrelid::pg_catalog.regclass, stxnamespace::pg_catalog.regnamespace::pg_catalog.text as nsp, stxname, pg_catalog.pg_get_statisticsobjdef_columns(oid) as columns, 'd' = any(stxkind) as ndist_enabled, 'f' = any(stxkind) as deps_enabled, 'm' = any(stxkind) as mcv_enabled, stxstattarget from pg_catalog.pg_statistic_ext where stxrelid = '";
    let suffix = "' order by nsp, stxname";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_inherits_parent_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select c.oid::pg_catalog.regclass from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhparent and i.inhrelid = '";
    let suffix = "' and c.relkind != 'p' and c.relkind != 'i' order by inhseqno";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_inherits_child_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select c.oid::pg_catalog.regclass, c.relkind, inhdetachpending, pg_catalog.pg_get_expr(c.relpartbound, c.oid) from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhrelid and i.inhparent = '";
    let suffix = "' order by pg_catalog.pg_get_expr(c.relpartbound, c.oid) = 'default', c.oid::pg_catalog.regclass::pg_catalog.text";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_empty_rows_for_relation_oid(_oid: u32) -> Vec<Vec<Option<String>>> {
    Vec::new()
}

fn catalog_empty_rows() -> Vec<Vec<Option<String>>> {
    Vec::new()
}

fn bootstrap_extension_description(session: &Session) -> String {
    session
        .comments
        .get(&CatalogCommentTarget::Extension {
            extension: "plpgsql".to_string(),
        })
        .cloned()
        .unwrap_or_else(|| PLPGSQL_DESCRIPTION.to_string())
}

fn catalog_psql_extension_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("plpgsql".to_string()),
        Some("1.0".to_string()),
        Some("pg_catalog".to_string()),
        Some(bootstrap_extension_description(session)),
    ]]
}

fn catalog_extension_discovery_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some(PG_EXTENSION_CLASS_OID.to_string()),
        Some(PLPGSQL_EXTENSION_OID.to_string()),
        Some("plpgsql".to_string()),
        Some("pg_catalog".to_string()),
        Some("f".to_string()),
        Some("1.0".to_string()),
        None,
        None,
    ]]
}

fn catalog_psql_language_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("plpgsql".to_string()),
        Some("postgres".to_string()),
        Some("t".to_string()),
        Some(PLPGSQL_DESCRIPTION.to_string()),
    ]]
}

fn pg_language_discovery_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("lanname"),
        bool_column("lanpltrusted"),
        int4_column("lanplcallfoid"),
        int4_column("laninline"),
        int4_column("lanvalidator"),
        text_column("lanacl"),
        text_column("acldefault"),
        int4_column("lanowner"),
    ]
}

fn pg_language_discovery_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some(PG_LANGUAGE_CLASS_OID.to_string()),
        Some(PLPGSQL_LANGUAGE_OID.to_string()),
        Some("plpgsql".to_string()),
        Some("t".to_string()),
        Some(PLPGSQL_CALL_HANDLER_OID.to_string()),
        Some(PLPGSQL_INLINE_HANDLER_OID.to_string()),
        Some(PLPGSQL_VALIDATOR_OID.to_string()),
        None,
        Some("postgres=U/postgres".to_string()),
        Some("10".to_string()),
    ]]
}

fn pg_catalog_tables_query() -> &'static str {
    "select schemaname, tablename, tableowner from pg_catalog.pg_tables where schemaname = 'public' order by tablename"
}

fn pg_catalog_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("postgres".to_string()),
            ]
        })
        .collect()
}

fn pg_catalog_indexes_query() -> &'static str {
    "select schemaname, tablename, indexname, indexdef from pg_catalog.pg_indexes where schemaname = 'public' order by tablename, indexname"
}

fn pg_catalog_indexes_without_schema_query() -> &'static str {
    "select tablename, indexname, indexdef from pg_catalog.pg_indexes where schemaname = 'public' order by tablename, indexname"
}

fn pg_catalog_index_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .map(|index| {
            (
                index.table.clone(),
                index.name.clone(),
                catalog_index_definition(index),
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    rows.into_iter()
        .map(|(table, index, definition)| {
            vec![
                Some("public".to_string()),
                Some(table),
                Some(index),
                Some(definition),
            ]
        })
        .collect()
}

fn pg_catalog_index_rows_without_schema(session: &Session) -> Vec<Vec<Option<String>>> {
    pg_catalog_index_rows(session)
        .into_iter()
        .map(|row| row.into_iter().skip(1).collect())
        .collect()
}

fn psql_describe_index_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .map(|index| {
            (
                index.name.clone(),
                vec![
                    Some("public".to_string()),
                    Some(index.name.clone()),
                    Some("index".to_string()),
                    Some("postgres".to_string()),
                    Some(index.table.clone()),
                ],
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    rows.into_iter().map(|(_, row)| row).collect()
}

fn psql_describe_index_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .map(|index| {
            (
                index.name.clone(),
                vec![
                    Some("public".to_string()),
                    Some(index.name.clone()),
                    Some("index".to_string()),
                    Some("postgres".to_string()),
                    Some(index.table.clone()),
                    Some("permanent".to_string()),
                    Some("btree".to_string()),
                    None,
                    session
                        .comments
                        .get(&CatalogCommentTarget::Index {
                            index: index.name.clone(),
                        })
                        .cloned(),
                ],
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    rows.into_iter().map(|(_, row)| row).collect()
}

fn pg_catalog_class_plain_tables_query() -> &'static str {
    "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relkind = 'r' order by c.relname"
}

fn pg_catalog_class_plain_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    pg_catalog_class_plain_table_rows_from_tables(tables)
}

fn pg_catalog_class_plain_tables_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname in (";
    let suffix = ") and c.relkind = 'r' order by c.relname";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn pg_catalog_class_plain_table_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    pg_catalog_class_plain_table_rows_from_tables(tables)
}

fn pg_catalog_class_plain_table_rows_from_tables(tables: Vec<&Table>) -> Vec<Vec<Option<String>>> {
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some(table.oid.to_string()),
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("r".to_string()),
                Some("p".to_string()),
            ]
        })
        .collect()
}

fn pg_catalog_class_sequences_query() -> &'static str {
    "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relkind = 's' order by c.relname"
}

fn pg_catalog_class_materialized_views_query() -> &'static str {
    "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relkind = 'm' order by c.relname"
}

fn pg_catalog_class_sequence_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    sequences
        .into_iter()
        .map(|sequence| {
            vec![
                Some(sequence.oid.to_string()),
                Some("public".to_string()),
                Some(sequence.name.clone()),
                Some("s".to_string()),
                Some("p".to_string()),
            ]
        })
        .collect()
}

fn pg_catalog_class_materialized_view_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut views = session.materialized_views.values().collect::<Vec<_>>();
    views.sort_by(|left, right| left.name.cmp(&right.name));
    views
        .into_iter()
        .map(|view| {
            vec![
                Some(view.oid.to_string()),
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("m".to_string()),
                Some("p".to_string()),
            ]
        })
        .collect()
}

fn information_schema_tables_query() -> &'static str {
    "select table_schema, table_name, table_type from information_schema.tables where table_schema = 'public' order by table_name"
}

fn information_schema_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("BASE TABLE".to_string()),
            ]
        })
        .collect()
}

fn information_schema_base_table_discovery_query() -> &'static str {
    "select table_schema, table_name from information_schema.tables where table_type = 'base table' and table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name"
}

fn information_schema_base_table_discovery_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| vec![Some("public".to_string()), Some(table.name.clone())])
        .collect()
}

fn information_schema_tables_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select table_schema, table_name, table_type from information_schema.tables where table_schema = 'public' and table_name in (";
    let suffix = ") order by table_name";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn information_schema_table_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("BASE TABLE".to_string()),
            ]
        })
        .collect()
}

fn information_schema_rich_tables_query() -> &'static str {
    "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_schema = 'public' order by table_name"
}

fn information_schema_rich_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(information_schema_rich_table_row)
        .collect()
}

fn information_schema_rich_tables_query_table(canonical: &str) -> Option<String> {
    let prefix = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_schema = 'public' and table_name = '";
    let suffix = "' order by table_name";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_rich_tables_catalog_query_table(canonical: &str) -> Option<String> {
    let suffix = "' order by table_name";
    let current_database_prefix = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_catalog = current_database() and table_schema = 'public' and table_name = '";
    if let Some(table) = canonical
        .strip_prefix(current_database_prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
    {
        return Some(table.to_string());
    }
    let literal_catalog_prefix = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_catalog = 'postgres' and table_schema = 'public' and table_name = '";
    canonical
        .strip_prefix(literal_catalog_prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_rich_table_rows_for_table(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    session
        .tables
        .get(table)
        .map(information_schema_rich_table_row)
        .into_iter()
        .collect()
}

fn information_schema_rich_table_row(table: &Table) -> Vec<Option<String>> {
    vec![
        Some("postgres".to_string()),
        Some("public".to_string()),
        Some(table.name.clone()),
        Some("BASE TABLE".to_string()),
        None,
        None,
        None,
        None,
        None,
        Some("YES".to_string()),
        Some("NO".to_string()),
        None,
    ]
}

fn information_schema_columns_query_table(canonical: &str) -> Option<String> {
    let prefix = "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_column_rows(session: &Session, table: &str) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(column.def.name.clone()),
                Some(column.attnum.to_string()),
                Some(column_type_display_name(column)),
            ]
        })
        .collect()
}

fn information_schema_all_columns_query() -> &'static str {
    "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_column_discovery_query() -> &'static str {
    "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name, ordinal_position"
}

fn information_schema_all_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    Some(column_type_display_name(column)),
                ]
            })
        })
        .collect()
}

fn information_schema_columns_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name in (";
    let suffix = ") order by table_name, ordinal_position";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn information_schema_column_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    Some(column_type_display_name(column)),
                ]
            })
        })
        .collect()
}

fn information_schema_column_details_query_table(canonical: &str) -> Option<String> {
    let prefix = "select column_name, data_type, is_nullable, column_default from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_column_detail_rows(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(column_type_display_name(column)),
                Some("YES".to_string()),
                column.def.default.as_ref().map(format_column_default_expr),
            ]
        })
        .collect()
}

fn information_schema_column_udt_query_table(canonical: &str) -> Option<String> {
    let prefix = "select column_name, data_type, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_column_udt_rows(session: &Session, table: &str) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            let (udt_schema, udt_name) = information_schema_udt_metadata(column);
            vec![
                Some(column.def.name.clone()),
                Some(column_type_display_name(column)),
                Some(udt_schema),
                Some(udt_name),
            ]
        })
        .collect()
}

fn information_schema_rich_columns_query() -> &'static str {
    "select table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_rich_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                let (udt_schema, udt_name) = information_schema_udt_metadata(column);
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    column.def.default.as_ref().map(format_column_default_expr),
                    Some("YES".to_string()),
                    Some(column_type_display_name(column)),
                    Some(udt_schema),
                    Some(udt_name),
                ]
            })
        })
        .collect()
}

fn information_schema_extended_columns_query() -> &'static str {
    "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_extended_columns_query_table(canonical: &str) -> Option<String> {
    let prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_extended_columns_catalog_query_table(canonical: &str) -> Option<String> {
    let suffix = "' order by ordinal_position";
    let current_database_prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = current_database() and table_schema = 'public' and table_name = '";
    if let Some(table) = canonical
        .strip_prefix(current_database_prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
    {
        return Some(table.to_string());
    }
    let literal_catalog_prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = 'postgres' and table_schema = 'public' and table_name = '";
    canonical
        .strip_prefix(literal_catalog_prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_extended_columns_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name in (";
    let suffix = ") order by table_name, ordinal_position";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn information_schema_extended_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(information_schema_extended_column_rows_for_catalog_table)
        .collect()
}

fn information_schema_extended_column_rows_for_table(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    information_schema_extended_column_rows_for_catalog_table(table).collect()
}

fn information_schema_extended_column_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(information_schema_extended_column_rows_for_catalog_table)
        .collect()
}

fn information_schema_extended_column_rows_for_catalog_table(
    table: &Table,
) -> impl Iterator<Item = Vec<Option<String>>> + '_ {
    table.columns.iter().map(|column| {
        let (numeric_precision, numeric_precision_radix, numeric_scale) =
            information_schema_numeric_metadata(column.def.ty);
        let (udt_schema, udt_name) = information_schema_udt_metadata(column);
        vec![
            Some("postgres".to_string()),
            Some("public".to_string()),
            Some(table.name.clone()),
            Some(column.def.name.clone()),
            Some(column.attnum.to_string()),
            column.def.default.as_ref().map(format_column_default_expr),
            Some("YES".to_string()),
            Some(column_type_display_name(column)),
            None,
            numeric_precision.map(|value| value.to_string()),
            numeric_precision_radix.map(|value| value.to_string()),
            numeric_scale.map(|value| value.to_string()),
            Some(udt_schema),
            Some(udt_name),
        ]
    })
}

fn information_schema_numeric_metadata(ty: SqlType) -> (Option<i32>, Option<i32>, Option<i32>) {
    match ty {
        SqlType::Int2 => (Some(16), Some(2), Some(0)),
        SqlType::Int4 => (Some(32), Some(2), Some(0)),
        SqlType::Int8 => (Some(64), Some(2), Some(0)),
        // For NUMERIC(p,s) PostgreSQL reports the declared precision/scale in radix 10.
        SqlType::Numeric { precision, scale } => {
            (Some(i32::from(precision)), Some(10), Some(i32::from(scale)))
        }
        SqlType::Bool => (None, None, None),
        SqlType::Text => (None, None, None),
        SqlType::Date => (None, None, None),
        SqlType::Timestamp => (None, None, None),
        SqlType::Uuid => (None, None, None),
    }
}

fn information_schema_schemata_query() -> &'static str {
    "select schema_name, schema_owner from information_schema.schemata where schema_name = 'public' order by schema_name"
}

fn information_schema_schemata_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    if !session.public_schema_exists {
        return Vec::new();
    }
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
    ]]
}

fn information_schema_table_constraints_query() -> &'static str {
    "select table_schema, table_name, constraint_name, constraint_type from information_schema.table_constraints where table_schema = 'public' order by table_name, constraint_name"
}

fn information_schema_table_constraint_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = catalog_constraint_entries(session)
        .into_iter()
        .map(|entry| {
            vec![
                Some("public".to_string()),
                Some(entry.index.table.clone()),
                Some(entry.index.name.clone()),
                Some(catalog_constraint_type(&entry.index).to_string()),
            ]
        })
        .collect::<Vec<_>>();
    for table in session.tables.values() {
        for constraint in &table.check_constraints {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.name.clone()),
                Some("CHECK".to_string()),
            ]);
        }
        for constraint in &table.foreign_keys {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.name.clone()),
                Some("FOREIGN KEY".to_string()),
            ]);
        }
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[2].cmp(&right[2])));
    rows
}

fn information_schema_key_column_usage_query() -> &'static str {
    "select table_schema, table_name, column_name, constraint_name, ordinal_position from information_schema.key_column_usage where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_key_column_usage_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = catalog_constraint_entries(session)
        .into_iter()
        .map(|entry| {
            vec![
                Some("public".to_string()),
                Some(entry.index.table.clone()),
                Some(entry.index.column.clone()),
                Some(entry.index.name.clone()),
                Some("1".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    for table in session.tables.values() {
        for constraint in &table.foreign_keys {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.column.clone()),
                Some(constraint.name.clone()),
                Some("1".to_string()),
            ]);
        }
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[4].cmp(&right[4])));
    rows
}

fn information_schema_views_query() -> &'static str {
    "select table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into, is_trigger_updatable, is_trigger_deletable, is_trigger_insertable from information_schema.views where table_schema = 'public' order by table_name"
}

fn information_schema_view_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .views
        .values()
        .map(|view| {
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some(view.name.clone()),
                Some(view.definition.clone()),
                Some("NONE".to_string()),
                Some("NO".to_string()),
                Some("NO".to_string()),
                Some("NO".to_string()),
                Some("NO".to_string()),
                Some("NO".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[2].cmp(&right[2]));
    rows
}

fn pg_catalog_views_query() -> &'static str {
    "select schemaname, viewname, viewowner, definition from pg_catalog.pg_views where schemaname = 'public' order by viewname"
}

fn pg_catalog_view_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .views
        .values()
        .map(|view| {
            vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("postgres".to_string()),
                Some(view.definition.clone()),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn psql_describe_function_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .functions
        .values()
        .map(|function| {
            vec![
                Some("public".to_string()),
                Some(function.name.clone()),
                Some(sql_type_display_name(function.return_type).to_string()),
                None,
                Some("func".to_string()),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn psql_describe_function_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .functions
        .values()
        .map(|function| {
            vec![
                Some("public".to_string()),
                Some(function.name.clone()),
                Some(sql_type_display_name(function.return_type).to_string()),
                None,
                Some("func".to_string()),
                Some("volatile".to_string()),
                Some("unsafe".to_string()),
                Some("postgres".to_string()),
                Some("invoker".to_string()),
                function_acl_display(&function.acl),
                Some("sql".to_string()),
                None,
                session
                    .comments
                    .get(&CatalogCommentTarget::Function {
                        function: function.name.clone(),
                    })
                    .cloned(),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[1].cmp(&right[1]));
    rows
}

fn pg_catalog_function_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .functions
        .values()
        .map(|function| {
            vec![
                Some(function.oid.to_string()),
                Some("public".to_string()),
                Some(function.name.clone()),
                Some(function.return_type.postgres_oid().to_string()),
                Some(sql_type_display_name(function.return_type).to_string()),
                Some(function.body.clone()),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left[2].cmp(&right[2]));
    rows
}

fn pg_catalog_function_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for function in session.functions.values() {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Function {
            function: function.name.clone(),
        }) {
            rows.push(vec![Some(function.name.clone()), Some(description.clone())]);
        }
    }
    rows.sort_by(|left, right| left[0].cmp(&right[0]));
    rows
}

fn pg_catalog_constraints_query() -> &'static str {
    "select n.nspname, c.relname, con.conname, con.contype from pg_catalog.pg_constraint con join pg_catalog.pg_class c on c.oid = con.conrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' order by c.relname, con.conname"
}

fn pg_catalog_constraint_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = catalog_constraint_entries(session)
        .into_iter()
        .map(|entry| {
            vec![
                Some("public".to_string()),
                Some(entry.index.table.clone()),
                Some(entry.index.name.clone()),
                Some(catalog_constraint_contype(&entry.index).to_string()),
            ]
        })
        .collect::<Vec<_>>();
    for table in session.tables.values() {
        for constraint in &table.check_constraints {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.name.clone()),
                Some("c".to_string()),
            ]);
        }
        for constraint in &table.foreign_keys {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(constraint.name.clone()),
                Some("f".to_string()),
            ]);
        }
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[2].cmp(&right[2])));
    rows
}

fn pg_catalog_attrdefs_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, pg_catalog.pg_get_expr(d.adbin, d.adrelid) as default_expr from pg_catalog.pg_attrdef d join pg_catalog.pg_class c on c.oid = d.adrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace join pg_catalog.pg_attribute a on a.attrelid = d.adrelid and a.attnum = d.adnum where n.nspname = 'public' order by c.relname, a.attnum"
}

fn pg_catalog_attrdef_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().filter_map(|column| {
                column.def.default.as_ref().map(|default| {
                    vec![
                        Some("public".to_string()),
                        Some(table.name.clone()),
                        Some(column.def.name.clone()),
                        Some(format_column_default_expr(default)),
                    ]
                })
            })
        })
        .collect()
}

fn pg_catalog_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','v') order by c.relname, d.objsubid"
}

fn pg_catalog_descriptions_with_sequences_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','v','s') order by c.relname, d.objsubid"
}

fn pg_catalog_descriptions_with_materialized_views_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','v','m','s') order by c.relname, d.objsubid"
}

fn pg_catalog_table_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind = 'r' order by c.relname, d.objsubid"
}

fn pg_catalog_constraint_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, con.conname, d.description from pg_catalog.pg_description d join pg_catalog.pg_constraint con on con.oid = d.objoid join pg_catalog.pg_class c on c.oid = con.conrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' order by c.relname, con.conname"
}

fn pg_catalog_table_index_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, c.relkind, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','i','v') order by c.relkind, c.relname, d.objsubid"
}

fn pg_catalog_table_index_sequence_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, c.relkind, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','i','v','s') order by c.relkind, c.relname, d.objsubid"
}

fn pg_catalog_table_index_sequence_matview_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, c.relkind, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','i','v','m','s') order by c.relkind, c.relname, d.objsubid"
}

fn pg_catalog_table_index_descriptions_without_views_query() -> &'static str {
    "select n.nspname, c.relname, c.relkind, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind in ('r','i') order by c.relkind, c.relname, d.objsubid"
}

fn pg_catalog_table_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    for table in tables {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Table {
            table: table.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                None,
                Some(description.clone()),
            ]);
        }
        for column in &table.columns {
            if let Some(description) = session.comments.get(&CatalogCommentTarget::Column {
                table: table.name.clone(),
                attnum: column.attnum,
            }) {
                rows.push(vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(description.clone()),
                ]);
            }
        }
    }
    rows
}

fn pg_catalog_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = pg_catalog_table_description_rows(session);
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in views {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::View {
            view: view.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in materialized_views {
        if let Some(description) = session
            .comments
            .get(&CatalogCommentTarget::MaterializedView {
                materialized_view: view.name.clone(),
            })
        {
            rows.push(vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    for sequence in sequences {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Sequence {
            sequence: sequence.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(sequence.name.clone()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    rows
}

fn pg_catalog_constraint_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for entry in catalog_constraint_entries(session) {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Constraint {
            table: entry.index.table.clone(),
            constraint: entry.index.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(entry.index.table),
                Some(entry.index.name),
                Some(description.clone()),
            ]);
        }
    }
    for table in session.tables.values() {
        for constraint in &table.check_constraints {
            if let Some(description) = session.comments.get(&CatalogCommentTarget::Constraint {
                table: table.name.clone(),
                constraint: constraint.name.clone(),
            }) {
                rows.push(vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(constraint.name.clone()),
                    Some(description.clone()),
                ]);
            }
        }
        for constraint in &table.foreign_keys {
            if let Some(description) = session.comments.get(&CatalogCommentTarget::Constraint {
                table: table.name.clone(),
                constraint: constraint.name.clone(),
            }) {
                rows.push(vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(constraint.name.clone()),
                    Some(description.clone()),
                ]);
            }
        }
    }
    rows.sort_by(|left, right| left[1].cmp(&right[1]).then_with(|| left[2].cmp(&right[2])));
    rows
}

fn pg_catalog_table_index_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut indexes = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .collect::<Vec<_>>();
    indexes.sort_by(|left, right| left.name.cmp(&right.name));
    for index in indexes {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Index {
            index: index.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(index.name.clone()),
                Some("i".to_string()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    for row in pg_catalog_description_rows(session) {
        let name = row[1].as_deref().unwrap_or_default();
        let relkind = if session.views.contains_key(name) {
            "v"
        } else if session.materialized_views.contains_key(name) {
            "m"
        } else if session.sequences.contains_key(name) {
            "s"
        } else {
            "r"
        };
        rows.push(vec![
            row[0].clone(),
            row[1].clone(),
            Some(relkind.to_string()),
            row[2].clone(),
            row[3].clone(),
        ]);
    }
    rows
}

fn pg_catalog_table_index_description_rows_without_views(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    let mut indexes = session
        .indexes
        .iter()
        .filter(|index| session.tables.contains_key(&index.table))
        .collect::<Vec<_>>();
    indexes.sort_by(|left, right| left.name.cmp(&right.name));
    for index in indexes {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Index {
            index: index.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(index.name.clone()),
                Some("i".to_string()),
                None,
                Some(description.clone()),
            ]);
        }
    }
    for row in pg_catalog_table_description_rows(session) {
        rows.push(vec![
            row[0].clone(),
            row[1].clone(),
            Some("r".to_string()),
            row[2].clone(),
            row[3].clone(),
        ]);
    }
    rows
}

fn pg_dump_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    if let Some(description) = session.comments.get(&CatalogCommentTarget::Extension {
        extension: "plpgsql".to_string(),
    }) {
        rows.push(vec![
            Some(description.clone()),
            Some(PG_EXTENSION_CLASS_OID.to_string()),
            Some(PLPGSQL_EXTENSION_OID.to_string()),
            Some("0".to_string()),
        ]);
    }
    if let Some(description) = session.comments.get(&CatalogCommentTarget::Schema {
        schema: "public".to_string(),
    }) {
        rows.push(vec![
            Some(description.clone()),
            Some("2615".to_string()),
            Some(PUBLIC_NAMESPACE_OID.to_string()),
            Some("0".to_string()),
        ]);
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by_key(|table| table.oid);
    for table in tables {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Table {
            table: table.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(table.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
        for column in &table.columns {
            if let Some(description) = session.comments.get(&CatalogCommentTarget::Column {
                table: table.name.clone(),
                attnum: column.attnum,
            }) {
                rows.push(vec![
                    Some(description.clone()),
                    Some("1259".to_string()),
                    Some(table.oid.to_string()),
                    Some(column.attnum.to_string()),
                ]);
            }
        }
    }
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by_key(|view| view.oid);
    for view in views {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::View {
            view: view.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(view.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by_key(|sequence| sequence.oid);
    for sequence in sequences {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Sequence {
            sequence: sequence.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(sequence.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by_key(|view| view.oid);
    for view in materialized_views {
        if let Some(description) = session
            .comments
            .get(&CatalogCommentTarget::MaterializedView {
                materialized_view: view.name.clone(),
            })
        {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(view.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut domains = session.domains.values().collect::<Vec<_>>();
    domains.sort_by_key(|domain| domain.oid);
    for domain in domains {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Domain {
            domain: domain.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some("1247".to_string()),
                Some(domain.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut functions = session.functions.values().collect::<Vec<_>>();
    functions.sort_by(|left, right| left.name.cmp(&right.name));
    for function in functions {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Function {
            function: function.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(function.name.clone()),
                Some("function".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut publications = session.publications.values().collect::<Vec<_>>();
    publications.sort_by_key(|publication| publication.oid);
    for publication in publications {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Publication {
            publication: publication.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some(PG_PUBLICATION_CLASS_OID.to_string()),
                Some(publication.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut subscriptions = session.subscriptions.values().collect::<Vec<_>>();
    subscriptions.sort_by_key(|subscription| subscription.oid);
    for subscription in subscriptions {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Subscription {
            subscription: subscription.name.clone(),
        }) {
            rows.push(vec![
                Some(description.clone()),
                Some(PG_SUBSCRIPTION_CLASS_OID.to_string()),
                Some(subscription.oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut index_comments = session
        .comments
        .iter()
        .filter_map(|(target, description)| match target {
            CatalogCommentTarget::Index { index } => Some((index, description)),
            CatalogCommentTarget::Database { .. }
            | CatalogCommentTarget::Role { .. }
            | CatalogCommentTarget::Schema { .. }
            | CatalogCommentTarget::Tablespace { .. }
            | CatalogCommentTarget::Table { .. }
            | CatalogCommentTarget::Column { .. }
            | CatalogCommentTarget::View { .. }
            | CatalogCommentTarget::MaterializedView { .. }
            | CatalogCommentTarget::Extension { .. }
            | CatalogCommentTarget::Function { .. }
            | CatalogCommentTarget::Sequence { .. }
            | CatalogCommentTarget::Domain { .. }
            | CatalogCommentTarget::Publication { .. }
            | CatalogCommentTarget::Subscription { .. }
            | CatalogCommentTarget::Constraint { .. } => None,
        })
        .collect::<Vec<_>>();
    index_comments.sort_by(|left, right| left.0.cmp(right.0));
    for (index, description) in index_comments {
        if let Some(oid) = catalog_index_oid(session, index) {
            rows.push(vec![
                Some(description.clone()),
                Some("1259".to_string()),
                Some(oid.to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    let mut constraint_comments = session
        .comments
        .iter()
        .filter_map(|(target, description)| match target {
            CatalogCommentTarget::Constraint { table, constraint } => {
                Some((table, constraint, description))
            }
            CatalogCommentTarget::Database { .. }
            | CatalogCommentTarget::Role { .. }
            | CatalogCommentTarget::Schema { .. }
            | CatalogCommentTarget::Tablespace { .. }
            | CatalogCommentTarget::Table { .. }
            | CatalogCommentTarget::Column { .. }
            | CatalogCommentTarget::View { .. }
            | CatalogCommentTarget::MaterializedView { .. }
            | CatalogCommentTarget::Extension { .. }
            | CatalogCommentTarget::Function { .. }
            | CatalogCommentTarget::Sequence { .. }
            | CatalogCommentTarget::Domain { .. }
            | CatalogCommentTarget::Publication { .. }
            | CatalogCommentTarget::Subscription { .. }
            | CatalogCommentTarget::Index { .. } => None,
        })
        .collect::<Vec<_>>();
    constraint_comments
        .sort_by(|left, right| left.0.cmp(right.0).then_with(|| left.1.cmp(right.1)));
    for (table_name, constraint_name, description) in constraint_comments {
        if let Some(entry) = catalog_constraint_entries(session)
            .into_iter()
            .find(|entry| &entry.index.table == table_name && &entry.index.name == constraint_name)
        {
            rows.push(vec![
                Some(description.clone()),
                Some("2606".to_string()),
                Some(catalog_constraint_oid(&entry).to_string()),
                Some("0".to_string()),
            ]);
        }
    }
    rows.sort_by(|left, right| {
        let left_key = (
            left[1]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
            left[2]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
            left[3]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
        );
        let right_key = (
            right[1]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
            right[2]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
            right[3]
                .as_deref()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    rows
}

fn psql_object_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    if let Some(description) = session.comments.get(&CatalogCommentTarget::Extension {
        extension: "plpgsql".to_string(),
    }) {
        rows.push(vec![
            Some("pg_catalog".to_string()),
            Some("plpgsql".to_string()),
            Some("extension".to_string()),
            Some(description.clone()),
        ]);
    }
    if let Some(description) = session.comments.get(&CatalogCommentTarget::Schema {
        schema: "public".to_string(),
    }) {
        rows.push(vec![
            Some("public".to_string()),
            Some("public".to_string()),
            Some("schema".to_string()),
            Some(description.clone()),
        ]);
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    for table in tables {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Table {
            table: table.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("table".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut views = session.views.values().collect::<Vec<_>>();
    views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in views {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::View {
            view: view.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("view".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut materialized_views = session.materialized_views.values().collect::<Vec<_>>();
    materialized_views.sort_by(|left, right| left.name.cmp(&right.name));
    for view in materialized_views {
        if let Some(description) = session
            .comments
            .get(&CatalogCommentTarget::MaterializedView {
                materialized_view: view.name.clone(),
            })
        {
            rows.push(vec![
                Some("public".to_string()),
                Some(view.name.clone()),
                Some("materialized view".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut sequences = session.sequences.values().collect::<Vec<_>>();
    sequences.sort_by(|left, right| left.name.cmp(&right.name));
    for sequence in sequences {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Sequence {
            sequence: sequence.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(sequence.name.clone()),
                Some("sequence".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut publications = session.publications.values().collect::<Vec<_>>();
    publications.sort_by(|left, right| left.name.cmp(&right.name));
    for publication in publications {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Publication {
            publication: publication.name.clone(),
        }) {
            rows.push(vec![
                None,
                Some(publication.name.clone()),
                Some("publication".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut subscriptions = session.subscriptions.values().collect::<Vec<_>>();
    subscriptions.sort_by(|left, right| left.name.cmp(&right.name));
    for subscription in subscriptions {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Subscription {
            subscription: subscription.name.clone(),
        }) {
            rows.push(vec![
                None,
                Some(subscription.name.clone()),
                Some("subscription".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    let mut constraints = catalog_constraint_entries(session);
    constraints.sort_by(|left, right| {
        left.index
            .table
            .cmp(&right.index.table)
            .then_with(|| left.index.name.cmp(&right.index.name))
    });
    for entry in constraints {
        if let Some(description) = session.comments.get(&CatalogCommentTarget::Constraint {
            table: entry.index.table.clone(),
            constraint: entry.index.name.clone(),
        }) {
            rows.push(vec![
                Some("public".to_string()),
                Some(entry.index.name),
                Some("table constraint".to_string()),
                Some(description.clone()),
            ]);
        }
    }
    rows
}

fn psql_list_object_descriptions_query(canonical: &str) -> bool {
    canonical.starts_with(
        "select distinct tt.nspname as \"schema\", tt.name as \"name\", tt.object as \"object\", d.description as \"description\" from ( select pgc.oid as oid, pgc.tableoid as tableoid",
    ) && canonical.contains("cast('table constraint' as pg_catalog.text) as object")
        && canonical.contains("cast('domain constraint' as pg_catalog.text) as object")
        && canonical.contains("cast('operator class' as pg_catalog.text) as object")
        && canonical.contains("cast('operator family' as pg_catalog.text) as object")
        && canonical.contains("cast('rule' as pg_catalog.text) as object")
        && canonical.contains("cast('trigger' as pg_catalog.text) as object")
        && canonical.contains("join pg_catalog.pg_description d on (tt.oid = d.objoid and tt.tableoid = d.classoid and d.objsubid = 0)")
        && canonical.ends_with("order by 1, 2, 3")
}

/// The full supported-type catalog ordered by OID. Test fixture documenting the
/// complete `(oid, typname, typlen)` listing (the live `pg_type` handlers filter to the
/// query's literal OID/name set via `catalog_type_rows_by_oid_in`/`_by_name_in`).
#[cfg(test)]
fn catalog_type_rows_by_oid() -> Vec<Vec<Option<String>>> {
    let mut types = SUPPORTED_SQL_TYPES;
    types.sort_by_key(|ty| ty.postgres_oid());
    types
        .into_iter()
        .map(|ty| {
            vec![
                Some(ty.postgres_oid().to_string()),
                Some(ty.catalog_name().to_string()),
                Some(ty.type_size().to_string()),
            ]
        })
        .collect()
}

/// The full supported-type catalog ordered by name (test fixture; see above).
#[cfg(test)]
fn catalog_type_rows_by_name() -> Vec<Vec<Option<String>>> {
    let mut types = SUPPORTED_SQL_TYPES;
    types.sort_by_key(|ty| ty.catalog_name());
    types
        .into_iter()
        .map(|ty| {
            vec![
                Some(ty.catalog_name().to_string()),
                Some(ty.postgres_oid().to_string()),
                Some(ty.type_size().to_string()),
            ]
        })
        .collect()
}

/// `(oid, typname, typlen)` rows for the supported types whose OID is in `oids`, in
/// OID order. Backs the hardcoded `pg_type WHERE oid IN (...)` introspection query so it
/// honors the literal OID set instead of dumping every supported type (a latent gap that
/// only surfaced once the supported-type set grew past int4/text).
fn catalog_type_rows_by_oid_in(oids: &[u32]) -> Vec<Vec<Option<String>>> {
    let mut types: Vec<SqlType> = SUPPORTED_SQL_TYPES
        .into_iter()
        .filter(|ty| oids.contains(&ty.postgres_oid()))
        .collect();
    types.sort_by_key(|ty| ty.postgres_oid());
    types
        .into_iter()
        .map(|ty| {
            vec![
                Some(ty.postgres_oid().to_string()),
                Some(ty.catalog_name().to_string()),
                Some(ty.type_size().to_string()),
            ]
        })
        .collect()
}

/// `(typname, oid, typlen)` rows for the supported types named in `names`, in name order.
/// Backs the hardcoded `pg_type WHERE typname IN (...)` introspection query.
fn catalog_type_rows_by_name_in(names: &[&str]) -> Vec<Vec<Option<String>>> {
    let mut types: Vec<SqlType> = SUPPORTED_SQL_TYPES
        .into_iter()
        .filter(|ty| names.contains(&ty.catalog_name()))
        .collect();
    types.sort_by_key(|ty| ty.catalog_name());
    types
        .into_iter()
        .map(|ty| {
            vec![
                Some(ty.catalog_name().to_string()),
                Some(ty.postgres_oid().to_string()),
                Some(ty.type_size().to_string()),
            ]
        })
        .collect()
}

fn catalog_attribute_query_table(canonical: &str) -> Option<String> {
    let prefix = "select attname, atttypid from pg_catalog.pg_attribute where attrelid = '";
    let suffix = "'::regclass and attnum > 0 order by attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn catalog_attribute_detail_query_table(canonical: &str) -> Option<String> {
    let prefix =
        "select attnum, attname, atttypid, attlen from pg_catalog.pg_attribute where attrelid = '";
    let suffix = "'::regclass and attnum > 0 order by attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn catalog_attribute_rows(session: &Session, table: &str) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.def.name.clone()),
                    Some(column_type_oid(session, column).to_string()),
                ]
            })
            .collect(),
    )
}

fn catalog_attribute_detail_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.attnum.to_string()),
                    Some(column.def.name.clone()),
                    Some(column_type_oid(session, column).to_string()),
                    Some(column_type_size(column).to_string()),
                ]
            })
            .collect(),
    )
}

fn pg_catalog_class_attribute_type_query_table(canonical: &str) -> Option<String> {
    let prefix = "select a.attnum, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type, a.attnotnull from pg_catalog.pg_attribute a join pg_catalog.pg_class c on c.oid = a.attrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn pg_catalog_class_attribute_type_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.attnum.to_string()),
                    Some(column.def.name.clone()),
                    Some(column_type_display_name(column)),
                    Some("f".to_string()),
                ]
            })
            .collect(),
    )
}

fn canonical_sql(input: &str) -> String {
    let mut sql = input.trim();
    while let Some(stripped) = sql.strip_suffix(';') {
        sql = stripped.trim_end();
    }

    let mut canonical = String::with_capacity(sql.len());
    let mut previous_was_space = false;
    for ch in sql.chars() {
        if ch.is_whitespace() {
            if !previous_was_space {
                canonical.push(' ');
                previous_was_space = true;
            }
        } else {
            canonical.extend(ch.to_lowercase());
            previous_was_space = false;
        }
    }
    canonical.trim().to_owned()
}

#[cfg(test)]
#[path = "gpu-db-server/tests.rs"]
mod tests;
