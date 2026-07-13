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
#[path = "gpu-db-server/pg_dump_relation_metadata.rs"]
mod pg_dump_relation_metadata;
#[cfg(test)]
use pg_dump_relation_metadata::test_pg_dump_index_metadata_rows as pg_dump_index_metadata_rows;
#[path = "gpu-db-server/pg_dump_compat.rs"]
mod pg_dump_compat;
use pg_dump_compat::try_execute_pg_dump_compat_statement;
#[path = "gpu-db-server/session_commands.rs"]
mod session_commands;
use session_commands::try_execute_session_command;
#[path = "gpu-db-server/bootstrap_ddl.rs"]
mod bootstrap_ddl;
#[cfg(test)]
use bootstrap_ddl::{
    test_catalog_psql_describe_schema_rows as catalog_psql_describe_schema_rows,
    test_catalog_psql_describe_schema_verbose_rows as catalog_psql_describe_schema_verbose_rows,
    test_catalog_psql_extension_rows as catalog_psql_extension_rows,
    test_catalog_psql_list_access_method_rows as catalog_psql_list_access_method_rows,
    test_information_schema_schemata_query as information_schema_schemata_query,
    test_information_schema_schemata_rows as information_schema_schemata_rows,
    test_is_pg_dump_public_namespace_oid_lookup_query as is_pg_dump_public_namespace_oid_lookup_query,
    test_pg_catalog_namespace_query as pg_catalog_namespace_query,
    test_pg_catalog_namespace_rows as pg_catalog_namespace_rows,
    test_psql_describe_schemas_catalog_query as psql_describe_schemas_catalog_query,
    test_psql_describe_schemas_verbose_catalog_query_public_filter as psql_describe_schemas_verbose_catalog_query_public_filter,
    test_psql_list_access_methods_catalog_query as psql_list_access_methods_catalog_query,
    test_psql_list_extensions_catalog_query as psql_list_extensions_catalog_query,
    test_psql_list_languages_catalog_query as psql_list_languages_catalog_query,
};
use bootstrap_ddl::{
    try_execute_access_method_catalog_query, try_execute_bootstrap_ddl,
    try_execute_extension_catalog_query, try_execute_information_schema_schemata_query,
    try_execute_language_catalog_query, try_execute_namespace_catalog_query,
    try_execute_psql_schema_catalog_query,
};
#[path = "gpu-db-server/cluster_ddl.rs"]
mod cluster_ddl;
use cluster_ddl::{
    execute_cluster_ddl, try_execute_database_catalog_query, try_execute_tablespace_catalog_query,
};
#[cfg(test)]
use cluster_ddl::{
    test_catalog_database_acl_rows as catalog_database_acl_rows,
    test_catalog_database_oid_rows as catalog_database_oid_rows,
    test_catalog_psql_list_database_rows as catalog_psql_list_database_rows,
    test_catalog_psql_list_database_verbose_rows as catalog_psql_list_database_verbose_rows,
    test_catalog_psql_list_tablespace_rows as catalog_psql_list_tablespace_rows,
    test_catalog_tablespace_acl_rows as catalog_tablespace_acl_rows,
    test_catalog_tablespace_oid_rows as catalog_tablespace_oid_rows,
    test_is_pg_dumpall_tablespace_metadata_query as is_pg_dumpall_tablespace_metadata_query,
    test_pg_dump_database_metadata_query as pg_dump_database_metadata_query,
    test_pg_dump_database_metadata_rows as pg_dump_database_metadata_rows,
    test_pg_dumpall_tablespace_metadata_rows as pg_dumpall_tablespace_metadata_rows,
    test_psql_list_databases_catalog_query as psql_list_databases_catalog_query,
    test_psql_list_databases_verbose_catalog_query as psql_list_databases_verbose_catalog_query,
    test_psql_list_tablespaces_catalog_query as psql_list_tablespaces_catalog_query,
    test_psql_list_tablespaces_verbose_catalog_query as psql_list_tablespaces_verbose_catalog_query,
};
#[path = "gpu-db-server/index_ddl.rs"]
mod index_ddl;
use index_ddl::{
    execute_index_ddl, try_execute_index_catalog_query, try_execute_index_direct_catalog_query,
};
#[cfg(test)]
use index_ddl::{
    rename_index_in_session, test_pg_catalog_index_rows as pg_catalog_index_rows,
    test_pg_catalog_indexes_query as pg_catalog_indexes_query,
    test_psql_describe_index_rows as psql_describe_index_rows,
    test_psql_describe_index_verbose_rows as psql_describe_index_verbose_rows,
    test_psql_describe_indexes_catalog_query as psql_describe_indexes_catalog_query,
    test_psql_describe_indexes_catalog_query_schema_filter as psql_describe_indexes_catalog_query_schema_filter,
};
#[path = "gpu-db-server/view_ddl.rs"]
mod view_ddl;
use view_ddl::{
    execute_view_ddl, try_execute_materialized_view_class_catalog_query,
    try_execute_view_catalog_query, try_execute_view_relation_catalog_query,
};
#[cfg(test)]
use view_ddl::{
    test_information_schema_view_rows as information_schema_view_rows,
    test_information_schema_views_query as information_schema_views_query,
    test_pg_catalog_view_rows as pg_catalog_view_rows,
    test_pg_catalog_views_query as pg_catalog_views_query,
    test_psql_describe_materialized_views_catalog_query as psql_describe_materialized_views_catalog_query,
    test_psql_describe_materialized_views_verbose_catalog_query as psql_describe_materialized_views_verbose_catalog_query,
    test_psql_describe_views_catalog_query as psql_describe_views_catalog_query,
    test_psql_describe_views_verbose_catalog_query as psql_describe_views_verbose_catalog_query,
};
#[path = "gpu-db-server/function_execution.rs"]
mod function_execution;
use function_execution::{
    execute_function_command, try_execute_aggregate_catalog_query,
    try_execute_function_catalog_query,
};
#[path = "gpu-db-server/type_system_catalog.rs"]
mod type_system_catalog;
#[cfg(test)]
use function_execution::{
    execute_function_result, rename_function_in_session,
    test_pg_catalog_function_rows as pg_catalog_function_rows,
    test_psql_describe_function_verbose_rows as psql_describe_function_verbose_rows,
    test_psql_describe_functions_catalog_query as psql_describe_functions_catalog_query,
    test_psql_list_aggregates_catalog_query as psql_list_aggregates_catalog_query,
};
#[cfg(test)]
use type_system_catalog::{
    test_catalog_psql_describe_type_rows as catalog_psql_describe_type_rows,
    test_catalog_psql_describe_type_rows_for_supported_types as catalog_psql_describe_type_rows_for_supported_types,
    test_catalog_psql_describe_type_verbose_rows_for_supported_types as catalog_psql_describe_type_verbose_rows_for_supported_types,
    test_catalog_type_rows_by_name as catalog_type_rows_by_name,
    test_catalog_type_rows_by_oid as catalog_type_rows_by_oid,
    test_psql_describe_pg_catalog_types_query as psql_describe_pg_catalog_types_query,
    test_psql_describe_query_type_rows as psql_describe_query_type_rows,
    test_psql_describe_type_catalog_query_type as psql_describe_type_catalog_query_type,
    test_psql_list_casts_catalog_query as psql_list_casts_catalog_query,
    test_psql_list_collations_catalog_query as psql_list_collations_catalog_query,
    test_psql_list_conversions_catalog_query as psql_list_conversions_catalog_query,
    test_psql_list_operators_catalog_query as psql_list_operators_catalog_query,
};
use type_system_catalog::{
    try_execute_builtin_type_catalog_query, try_execute_direct_type_catalog_query,
    try_execute_psql_describe_result_type_catalog_query, try_execute_type_system_catalog_query,
};
#[path = "gpu-db-server/table_catalog.rs"]
mod table_catalog;
#[cfg(test)]
use table_catalog::{
    test_catalog_psql_describe_table_privilege_rows_filtered as catalog_psql_describe_table_privilege_rows_filtered,
    test_catalog_psql_describe_table_rows as catalog_psql_describe_table_rows,
    test_catalog_psql_describe_table_rows_filtered as catalog_psql_describe_table_rows_filtered,
    test_catalog_psql_describe_table_verbose_rows as catalog_psql_describe_table_verbose_rows,
    test_catalog_psql_describe_table_verbose_rows_filtered as catalog_psql_describe_table_verbose_rows_filtered,
    test_catalog_table_name_rows as catalog_table_name_rows,
    test_catalog_table_oid_rows as catalog_table_oid_rows,
    test_information_schema_base_table_discovery_query as information_schema_base_table_discovery_query,
    test_information_schema_base_table_discovery_rows as information_schema_base_table_discovery_rows,
    test_information_schema_rich_table_rows as information_schema_rich_table_rows,
    test_information_schema_rich_table_rows_for_table as information_schema_rich_table_rows_for_table,
    test_information_schema_rich_tables_catalog_query_table as information_schema_rich_tables_catalog_query_table,
    test_information_schema_rich_tables_query as information_schema_rich_tables_query,
    test_information_schema_rich_tables_query_table as information_schema_rich_tables_query_table,
    test_information_schema_table_rows as information_schema_table_rows,
    test_information_schema_table_rows_for_tables as information_schema_table_rows_for_tables,
    test_information_schema_tables_in_query_tables as information_schema_tables_in_query_tables,
    test_pg_catalog_class_plain_table_rows as pg_catalog_class_plain_table_rows,
    test_pg_catalog_class_plain_table_rows_for_tables as pg_catalog_class_plain_table_rows_for_tables,
    test_pg_catalog_class_plain_tables_in_query_tables as pg_catalog_class_plain_tables_in_query_tables,
    test_pg_catalog_class_plain_tables_query as pg_catalog_class_plain_tables_query,
    test_pg_catalog_table_rows as pg_catalog_table_rows,
    test_pg_catalog_tables_query as pg_catalog_tables_query,
    test_psql_describe_all_schema_tables_catalog_query as psql_describe_all_schema_tables_catalog_query,
    test_psql_describe_all_schema_tables_verbose_catalog_query as psql_describe_all_schema_tables_verbose_catalog_query,
    test_psql_describe_relations_catalog_query as psql_describe_relations_catalog_query,
    test_psql_describe_table_privileges_catalog_query_filter as psql_describe_table_privileges_catalog_query_filter,
    test_psql_describe_tables_catalog_query_filter as psql_describe_tables_catalog_query_filter,
    test_psql_describe_tables_verbose_catalog_query as psql_describe_tables_verbose_catalog_query,
    test_psql_describe_tables_verbose_catalog_query_filter as psql_describe_tables_verbose_catalog_query_filter,
    test_relation_acl_display as relation_acl_display, PsqlDescribeTablesFilter,
};
use table_catalog::{
    try_execute_filtered_plain_table_class_catalog_query,
    try_execute_information_schema_table_catalog_query, try_execute_pg_catalog_tables_query,
    try_execute_plain_table_class_catalog_query, try_execute_table_catalog_query,
};
#[path = "gpu-db-server/column_catalog.rs"]
mod column_catalog;
#[cfg(test)]
use column_catalog::{
    test_catalog_attribute_detail_query_table as catalog_attribute_detail_query_table,
    test_catalog_attribute_detail_rows as catalog_attribute_detail_rows,
    test_catalog_attribute_query_table as catalog_attribute_query_table,
    test_catalog_attribute_rows as catalog_attribute_rows,
    test_information_schema_all_column_rows as information_schema_all_column_rows,
    test_information_schema_all_columns_query as information_schema_all_columns_query,
    test_information_schema_column_detail_rows as information_schema_column_detail_rows,
    test_information_schema_column_details_query_table as information_schema_column_details_query_table,
    test_information_schema_column_discovery_query as information_schema_column_discovery_query,
    test_information_schema_column_rows as information_schema_column_rows,
    test_information_schema_column_rows_for_tables as information_schema_column_rows_for_tables,
    test_information_schema_columns_in_query_tables as information_schema_columns_in_query_tables,
    test_information_schema_columns_query_table as information_schema_columns_query_table,
    test_information_schema_extended_column_rows as information_schema_extended_column_rows,
    test_information_schema_extended_column_rows_for_table as information_schema_extended_column_rows_for_table,
    test_information_schema_extended_column_rows_for_tables as information_schema_extended_column_rows_for_tables,
    test_information_schema_extended_columns_catalog_query_table as information_schema_extended_columns_catalog_query_table,
    test_information_schema_extended_columns_in_query_tables as information_schema_extended_columns_in_query_tables,
    test_information_schema_extended_columns_query as information_schema_extended_columns_query,
    test_information_schema_extended_columns_query_table as information_schema_extended_columns_query_table,
    test_information_schema_numeric_metadata as information_schema_numeric_metadata,
    test_information_schema_rich_column_rows as information_schema_rich_column_rows,
    test_information_schema_rich_columns_query as information_schema_rich_columns_query,
    test_pg_catalog_class_attribute_type_query_table as pg_catalog_class_attribute_type_query_table,
    test_pg_catalog_class_attribute_type_rows as pg_catalog_class_attribute_type_rows,
};
use column_catalog::{
    try_execute_direct_attribute_catalog_query, try_execute_information_schema_column_catalog_query,
};
#[path = "gpu-db-server/constraint_catalog.rs"]
mod constraint_catalog;
#[cfg(test)]
use constraint_catalog::{
    test_information_schema_key_column_usage_query as information_schema_key_column_usage_query,
    test_information_schema_key_column_usage_rows as information_schema_key_column_usage_rows,
    test_information_schema_table_constraint_rows as information_schema_table_constraint_rows,
    test_information_schema_table_constraints_query as information_schema_table_constraints_query,
    test_pg_catalog_attrdef_rows as pg_catalog_attrdef_rows,
    test_pg_catalog_attrdefs_query as pg_catalog_attrdefs_query,
    test_pg_catalog_constraint_rows as pg_catalog_constraint_rows,
    test_pg_catalog_constraints_query as pg_catalog_constraints_query,
};
use constraint_catalog::{
    try_execute_information_schema_constraint_catalog_query,
    try_execute_pg_constraint_catalog_query,
};
#[path = "gpu-db-server/relation_introspection.rs"]
mod relation_introspection;
use relation_introspection::try_execute_relation_introspection_query;
#[cfg(test)]
use relation_introspection::{
    test_catalog_describe_attribute_query_oid as catalog_describe_attribute_query_oid,
    test_catalog_describe_attribute_rows as catalog_describe_attribute_rows,
    test_catalog_describe_inherits_child_query_oid as catalog_describe_inherits_child_query_oid,
    test_catalog_describe_inherits_parent_query_oid as catalog_describe_inherits_parent_query_oid,
    test_catalog_describe_policy_query_oid as catalog_describe_policy_query_oid,
    test_catalog_describe_relation_flags_query_oid as catalog_describe_relation_flags_query_oid,
    test_catalog_describe_relation_flags_rows as catalog_describe_relation_flags_rows,
    test_catalog_describe_relation_lookup_query_all_schemas as catalog_describe_relation_lookup_query_all_schemas,
    test_catalog_describe_relation_lookup_query_public_namespace as catalog_describe_relation_lookup_query_public_namespace,
    test_catalog_describe_relation_lookup_query_table as catalog_describe_relation_lookup_query_table,
    test_catalog_describe_relation_lookup_rows as catalog_describe_relation_lookup_rows,
    test_catalog_describe_relation_lookup_rows_for_public_namespace as catalog_describe_relation_lookup_rows_for_public_namespace,
    test_catalog_describe_statistic_ext_query_oid as catalog_describe_statistic_ext_query_oid,
    test_catalog_describe_verbose_attribute_query_oid as catalog_describe_verbose_attribute_query_oid,
    test_catalog_describe_verbose_attribute_rows as catalog_describe_verbose_attribute_rows,
};
#[path = "gpu-db-server/sequence_execution.rs"]
mod sequence_execution;
use sequence_execution::{
    create_implicit_sequence, execute_sequence_command, next_sequence_value, sequence_target_error,
    try_execute_sequence_catalog_query, try_execute_sequence_class_catalog_query,
};
#[cfg(test)]
use sequence_execution::{
    rename_sequence_in_session,
    test_pg_catalog_class_sequence_rows as pg_catalog_class_sequence_rows,
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
use role_ddl::{execute_role_ddl, try_execute_role_catalog_query};
#[cfg(test)]
use role_ddl::{
    test_catalog_psql_describe_role_rows as catalog_psql_describe_role_rows,
    test_psql_describe_roles_catalog_query as psql_describe_roles_catalog_query,
    test_psql_describe_roles_verbose_catalog_query as psql_describe_roles_verbose_catalog_query,
};
#[path = "gpu-db-server/catalog_comments.rs"]
mod catalog_comments;
use catalog_comments::{
    execute_catalog_comment, try_execute_relation_description_catalog_query,
    try_execute_schema_description_catalog_query,
};
#[cfg(test)]
use catalog_comments::{
    test_pg_catalog_constraint_description_rows as pg_catalog_constraint_description_rows,
    test_pg_catalog_description_rows as pg_catalog_description_rows,
    test_pg_catalog_schema_description_rows as pg_catalog_schema_description_rows,
    test_pg_catalog_table_description_rows as pg_catalog_table_description_rows,
    test_pg_catalog_table_descriptions_query as pg_catalog_table_descriptions_query,
    test_pg_dump_description_rows as pg_dump_description_rows,
    test_psql_list_object_descriptions_query as psql_list_object_descriptions_query,
    test_psql_object_description_rows as psql_object_description_rows,
};
#[path = "gpu-db-server/acl_execution.rs"]
mod acl_execution;
use acl_execution::{
    acl_display, database_acl_display, database_exists, execute_acl_command,
    function_access_permission_error, function_acl_array_display, function_privilege_letters,
    object_access_permission_error, relation_acl_array_display, role_exists, role_has_dependencies,
    schema_acl_array_display, schema_acl_display, schema_permission_error,
    schema_usage_permission_error, tablespace_acl_array_display, tablespace_acl_display,
    tablespace_exists, try_execute_default_acl_catalog_query,
};
#[cfg(test)]
use acl_execution::{
    grant_default_table_acl, grant_function_acl, grant_relation_acl, grant_schema_acl,
    revoke_relation_acl,
    test_catalog_psql_default_access_privilege_rows as catalog_psql_default_access_privilege_rows,
    test_psql_list_default_access_privileges_catalog_query as psql_list_default_access_privileges_catalog_query,
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

    if let Some(result) = try_execute_psql_describe_result_type_catalog_query(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_table_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_index_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_view_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_sequence_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_function_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_aggregate_catalog_query(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_type_system_catalog_query(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_replication_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_default_acl_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_extension_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_language_catalog_query(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_domain_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_role_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_database_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_tablespace_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_access_method_catalog_query(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_psql_schema_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_schema_description_catalog_query(stream, session, &canonical)
    {
        return result;
    }
    if let Some(result) = try_execute_namespace_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_builtin_type_catalog_query(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_relation_introspection_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_pg_catalog_tables_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_index_direct_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_plain_table_class_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_sequence_class_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) =
        try_execute_materialized_view_class_catalog_query(stream, session, &canonical)
    {
        return result;
    }
    if let Some(result) =
        try_execute_filtered_plain_table_class_catalog_query(stream, session, &canonical)
    {
        return result;
    }
    if let Some(result) =
        try_execute_information_schema_table_catalog_query(stream, session, &canonical)
    {
        return result;
    }
    if let Some(result) =
        try_execute_information_schema_column_catalog_query(stream, session, &canonical)
    {
        return result;
    }
    if let Some(result) = try_execute_information_schema_schemata_query(stream, session, &canonical)
    {
        return result;
    }
    if let Some(result) =
        try_execute_information_schema_constraint_catalog_query(stream, session, &canonical)
    {
        return result;
    }
    if let Some(result) = try_execute_view_relation_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_pg_constraint_catalog_query(stream, session, &canonical) {
        return result;
    }
    if let Some(result) =
        try_execute_relation_description_catalog_query(stream, session, &canonical)
    {
        return result;
    }
    if let Some(result) = try_execute_direct_type_catalog_query(stream, &canonical) {
        return result;
    }
    if let Some(result) = try_execute_direct_attribute_catalog_query(stream, session, &canonical) {
        return result;
    }
    execute_session_compat_fallback(stream, session, &canonical)
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

fn foreign_key_definition(foreign_key: &CatalogForeignKey) -> String {
    format!(
        "FOREIGN KEY ({}) REFERENCES {}({})",
        foreign_key.column, foreign_key.referenced_table, foreign_key.referenced_column
    )
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

fn catalog_empty_rows() -> Vec<Vec<Option<String>>> {
    Vec::new()
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
