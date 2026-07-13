// Legacy pg_dump compatibility ownership. This is not a product catalog execution path.

use super::bootstrap_ddl::{
    try_execute_extension_pg_dump_catalog_query, try_execute_language_pg_dump_catalog_query,
    try_execute_schema_pg_dump_catalog_query,
};
use super::catalog_comments::pg_dump_description_rows;
use super::cluster_ddl::{
    try_execute_database_pg_dump_catalog_query, try_execute_tablespace_pg_dump_catalog_query,
};
use super::replication_catalog::try_execute_replication_pg_dump_query;
use super::role_ddl::try_execute_role_pg_dump_catalog_query;
use super::{
    bool_column, catalog_empty_rows, catalog_foreign_key_metadata_columns,
    catalog_foreign_key_metadata_rows, int4_column, int8_column,
    is_catalog_foreign_key_metadata_query, is_pg_dump_class_metadata_query,
    is_pg_dump_default_acl_metadata_query, is_pg_dump_function_metadata_query,
    is_pg_dump_index_metadata_query, pg_dump_attrdef_metadata_columns,
    pg_dump_attrdef_metadata_query_relation_oids, pg_dump_attrdef_metadata_rows,
    pg_dump_attribute_metadata_columns, pg_dump_attribute_metadata_query_oids,
    pg_dump_attribute_metadata_rows, pg_dump_class_metadata_columns, pg_dump_class_metadata_rows,
    pg_dump_default_acl_metadata_columns, pg_dump_default_acl_metadata_rows,
    pg_dump_dependency_rows, pg_dump_empty_catalog_query_columns,
    pg_dump_function_metadata_columns, pg_dump_function_metadata_rows,
    pg_dump_index_metadata_columns, pg_dump_index_metadata_rows,
    pg_dump_sequence_last_value_query_name, pg_dump_sequence_metadata_columns,
    pg_dump_sequence_metadata_query_oid, pg_dump_sequence_metadata_rows,
    pg_dump_sequence_setval_query, pg_dump_table_oid_lookup_query_table,
    pg_dump_table_oid_lookup_rows, pg_dump_type_metadata_columns, pg_dump_type_metadata_query,
    pg_dump_type_metadata_rows, pg_dump_view_definition_query_oid, pg_dump_view_definition_rows,
    text_column, write_error, write_single_row, CatalogCommentTarget, ErrorField, ReadWrite,
    Session,
};
use std::io;

pub(super) fn try_execute_pg_dump_compat_statement(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    canonical: &str,
) -> Option<io::Result<()>> {
    if canonical
        == "select count(*) from pg_subscription where subdbid = (select oid from pg_database where datname = current_database())"
    {
        return Some(write_single_row(
            stream,
            &[int4_column("count")],
            &[vec![Some(session.subscriptions.len().to_string())]],
        ));
    }
    if let Some(result) = try_execute_role_pg_dump_catalog_query(stream, session, canonical) {
        return Some(result);
    }
    if let Some(result) = try_execute_extension_pg_dump_catalog_query(stream, canonical) {
        return Some(result);
    }
    if let Some(result) = try_execute_language_pg_dump_catalog_query(stream, canonical) {
        return Some(result);
    }
    if let Some(result) = try_execute_schema_pg_dump_catalog_query(stream, session, canonical) {
        return Some(result);
    }
    if let Some(table) = pg_dump_table_oid_lookup_query_table(canonical) {
        return Some(write_single_row(
            stream,
            &[int4_column("oid")],
            &pg_dump_table_oid_lookup_rows(session, &table),
        ));
    }
    if is_pg_dump_class_metadata_query(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_class_metadata_columns(),
            &pg_dump_class_metadata_rows(session),
        ));
    }
    if let Some(oids) = pg_dump_attribute_metadata_query_oids(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_attribute_metadata_columns(),
            &pg_dump_attribute_metadata_rows(session, &oids),
        ));
    }
    if canonical == pg_dump_type_metadata_query() {
        return Some(write_single_row(
            stream,
            &pg_dump_type_metadata_columns(),
            &pg_dump_type_metadata_rows(session),
        ));
    }
    if let Some(result) = try_execute_database_pg_dump_catalog_query(stream, session, canonical) {
        return Some(result);
    }
    if let Some(sequence_oid) = pg_dump_sequence_metadata_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_sequence_metadata_columns(),
            &pg_dump_sequence_metadata_rows(session, sequence_oid),
        ));
    }
    if canonical == "select pg_catalog.shobj_description(5, 'pg_database')" {
        return Some(write_single_row(
            stream,
            &[text_column("shobj_description")],
            &[vec![session
                .comments
                .get(&CatalogCommentTarget::Database {
                    database: "postgres".to_string(),
                })
                .cloned()]],
        ));
    }
    if canonical == "select pg_catalog.shobj_description(10, 'pg_authid')" {
        return Some(write_single_row(
            stream,
            &[text_column("shobj_description")],
            &[vec![session
                .comments
                .get(&CatalogCommentTarget::Role {
                    role: "postgres".to_string(),
                })
                .cloned()]],
        ));
    }
    if canonical == "select pg_catalog.shobj_description(10, 'pg_authid') as role_comment" {
        return Some(write_single_row(
            stream,
            &[text_column("role_comment")],
            &[vec![session
                .comments
                .get(&CatalogCommentTarget::Role {
                    role: "postgres".to_string(),
                })
                .cloned()]],
        ));
    }
    if canonical
        == "select pg_catalog.shobj_description(10, 'pg_authid') is null as role_comment_cleared"
    {
        return Some(write_single_row(
            stream,
            &[bool_column("role_comment_cleared")],
            &[vec![Some(
                if session.comments.contains_key(&CatalogCommentTarget::Role {
                    role: "postgres".to_string(),
                }) {
                    "f"
                } else {
                    "t"
                }
                .to_string(),
            )]],
        ));
    }
    if canonical == "select pg_catalog.shobj_description(1663, 'pg_tablespace')" {
        return Some(write_single_row(
            stream,
            &[text_column("shobj_description")],
            &[vec![session
                .comments
                .get(&CatalogCommentTarget::Tablespace {
                    tablespace: "pg_default".to_string(),
                })
                .cloned()]],
        ));
    }
    if canonical
        == "select pg_catalog.shobj_description(1663, 'pg_tablespace') as tablespace_comment"
    {
        return Some(write_single_row(
            stream,
            &[text_column("tablespace_comment")],
            &[vec![session
                .comments
                .get(&CatalogCommentTarget::Tablespace {
                    tablespace: "pg_default".to_string(),
                })
                .cloned()]],
        ));
    }
    if canonical
        == "select pg_catalog.shobj_description(1663, 'pg_tablespace') is null as tablespace_comment_cleared"
    {
        return Some(write_single_row(
            stream,
            &[bool_column("tablespace_comment_cleared")],
            &[vec![Some(
                if session
                    .comments
                    .contains_key(&CatalogCommentTarget::Tablespace {
                        tablespace: "pg_default".to_string(),
                    })
                {
                    "f"
                } else {
                    "t"
                }
                .to_string(),
            )]],
        ));
    }
    if canonical == "select pg_catalog.shobj_description(1664, 'pg_tablespace')" {
        return Some(write_single_row(
            stream,
            &[text_column("shobj_description")],
            &[vec![session
                .comments
                .get(&CatalogCommentTarget::Tablespace {
                    tablespace: "pg_global".to_string(),
                })
                .cloned()]],
        ));
    }
    if canonical
        == "select pg_catalog.shobj_description(1664, 'pg_tablespace') as global_tablespace_comment"
    {
        return Some(write_single_row(
            stream,
            &[text_column("global_tablespace_comment")],
            &[vec![session
                .comments
                .get(&CatalogCommentTarget::Tablespace {
                    tablespace: "pg_global".to_string(),
                })
                .cloned()]],
        ));
    }
    if canonical
        == "select pg_catalog.shobj_description(1664, 'pg_tablespace') is null as global_tablespace_comment_cleared"
    {
        return Some(write_single_row(
            stream,
            &[bool_column("global_tablespace_comment_cleared")],
            &[vec![Some(
                if session
                    .comments
                    .contains_key(&CatalogCommentTarget::Tablespace {
                        tablespace: "pg_global".to_string(),
                    })
                {
                    "f"
                } else {
                    "t"
                }
                .to_string(),
            )]],
        ));
    }
    if is_pg_dump_index_metadata_query(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_index_metadata_columns(),
            &pg_dump_index_metadata_rows(session),
        ));
    }
    if is_catalog_foreign_key_metadata_query(canonical) {
        return Some(write_single_row(
            stream,
            &catalog_foreign_key_metadata_columns(),
            &catalog_foreign_key_metadata_rows(session),
        ));
    }
    if let Some(view_oid) = pg_dump_view_definition_query_oid(canonical) {
        return Some(write_single_row(
            stream,
            &[text_column("viewdef")],
            &pg_dump_view_definition_rows(session, view_oid),
        ));
    }
    if let Some(relation_oids) = pg_dump_attrdef_metadata_query_relation_oids(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_attrdef_metadata_columns(),
            &pg_dump_attrdef_metadata_rows(session, &relation_oids),
        ));
    }
    if let Some(result) = try_execute_replication_pg_dump_query(stream, session, canonical) {
        return Some(result);
    }
    if is_pg_dump_function_metadata_query(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_function_metadata_columns(),
            &pg_dump_function_metadata_rows(session),
        ));
    }
    if let Some(result) = try_execute_tablespace_pg_dump_catalog_query(stream, session, canonical) {
        return Some(result);
    }
    if let Some(columns) = pg_dump_empty_catalog_query_columns(canonical) {
        return Some(write_single_row(stream, &columns, &catalog_empty_rows()));
    }
    if is_pg_dump_default_acl_metadata_query(canonical) {
        return Some(write_single_row(
            stream,
            &pg_dump_default_acl_metadata_columns(),
            &pg_dump_default_acl_metadata_rows(session),
        ));
    }
    if canonical.starts_with("with recursive w as ( select d1.objid") {
        return Some(write_single_row(
            stream,
            &[
                int4_column("classid"),
                int4_column("objid"),
                int4_column("refobjid"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical
        == "select classid, objid, refobjid from pg_depend where refclassid = 'pg_extension'::regclass and deptype = 'e' order by 3"
    {
        return Some(write_single_row(
            stream,
            &[
                int4_column("classid"),
                int4_column("objid"),
                int4_column("refobjid"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical
        == "select conrelid, confrelid from pg_constraint join pg_depend on (objid = confrelid) where contype = 'f' and refclassid = 'pg_extension'::regclass and classid = 'pg_class'::regclass"
    {
        return Some(write_single_row(
            stream,
            &[int4_column("conrelid"), int4_column("confrelid")],
            &catalog_empty_rows(),
        ));
    }
    if canonical.starts_with("select classid, objid, refclassid, refobjid, deptype from pg_depend")
    {
        return Some(write_single_row(
            stream,
            &[
                int4_column("classid"),
                int4_column("objid"),
                int4_column("refclassid"),
                int4_column("refobjid"),
                text_column("deptype"),
            ],
            &pg_dump_dependency_rows(session),
        ));
    }
    if canonical
        == "select description, classoid, objoid, objsubid from pg_catalog.pg_description order by classoid, objoid, objsubid"
    {
        return Some(write_single_row(
            stream,
            &[
                text_column("description"),
                int4_column("classoid"),
                int4_column("objoid"),
                int4_column("objsubid"),
            ],
            &pg_dump_description_rows(session),
        ));
    }
    if canonical
        == "select label, provider, classoid, objoid, objsubid from pg_catalog.pg_seclabels order by classoid, objoid, objsubid"
    {
        return Some(write_single_row(
            stream,
            &[
                text_column("label"),
                text_column("provider"),
                int4_column("classoid"),
                int4_column("objoid"),
                int4_column("objsubid"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical.starts_with("select provider, label from pg_catalog.pg_shseclabel where ") {
        return Some(write_single_row(
            stream,
            &[text_column("provider"), text_column("label")],
            &catalog_empty_rows(),
        ));
    }
    if canonical.starts_with("select unnest(setconfig) from pg_db_role_setting ")
        || canonical.starts_with("select rolname, unnest(setconfig) from pg_db_role_setting ")
    {
        let columns = if canonical.starts_with("select rolname,") {
            vec![text_column("rolname"), text_column("unnest")]
        } else {
            vec![text_column("unnest")]
        };
        return Some(write_single_row(stream, &columns, &catalog_empty_rows()));
    }
    if canonical.starts_with("select ur.rolname as role, um.rolname as member")
        && canonical.contains("from pg_auth_members a")
    {
        return Some(write_single_row(
            stream,
            &[
                text_column("role"),
                text_column("member"),
                text_column("grantor"),
                int4_column("roleid"),
                int4_column("memberid"),
                int4_column("grantorid"),
                bool_column("admin_option"),
                bool_column("inherit_option"),
                bool_column("set_option"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if canonical.starts_with("select parname, pg_catalog.pg_get_userbyid(10) as parowner")
        && canonical.contains("from pg_catalog.pg_parameter_acl")
    {
        return Some(write_single_row(
            stream,
            &[
                text_column("parname"),
                text_column("parowner"),
                text_column("paracl"),
                text_column("acldefault"),
            ],
            &catalog_empty_rows(),
        ));
    }
    if let Some((sequence, value, is_called)) = pg_dump_sequence_setval_query(canonical) {
        if let Some(sequence_state) = session.sequences.get_mut(&sequence) {
            sequence_state.last_value = value;
            sequence_state.is_called = is_called;
            session.mark_sequence_dirty(sequence);
            session.persist_catalog_snapshot();
            return Some(write_single_row(
                stream,
                &[int8_column("setval")],
                &[vec![Some(value.to_string())]],
            ));
        }
        return Some(write_error(
            stream,
            &ErrorField {
                code: "42P01",
                message: "relation does not exist",
                position: None,
            },
        ));
    }
    if let Some(sequence) = pg_dump_sequence_last_value_query_name(canonical) {
        if let Some(sequence_state) = session.sequences.get(&sequence) {
            return Some(write_single_row(
                stream,
                &[int8_column("last_value"), bool_column("is_called")],
                &[vec![
                    Some(sequence_state.last_value.to_string()),
                    Some(if sequence_state.is_called { "t" } else { "f" }.to_string()),
                ]],
            ));
        }
        return Some(write_error(
            stream,
            &ErrorField {
                code: "42P01",
                message: "relation does not exist",
                position: None,
            },
        ));
    }

    None
}
