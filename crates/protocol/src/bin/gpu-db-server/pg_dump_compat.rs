// Legacy pg_dump compatibility ownership. This is not a product catalog execution path.

use super::acl_execution::try_execute_default_acl_pg_dump_catalog_query;
use super::bootstrap_ddl::{
    try_execute_access_method_pg_dump_catalog_query, try_execute_extension_pg_dump_catalog_query,
    try_execute_language_pg_dump_catalog_query, try_execute_schema_pg_dump_catalog_query,
};
use super::catalog_comments::pg_dump_description_rows;
use super::cluster_ddl::{
    try_execute_database_pg_dump_catalog_query, try_execute_tablespace_pg_dump_catalog_query,
};
use super::function_execution::try_execute_aggregate_pg_dump_catalog_query;
use super::pg_dump_relation_metadata::{
    catalog_foreign_key_metadata_columns, catalog_foreign_key_metadata_rows,
    is_catalog_foreign_key_metadata_query, is_pg_dump_class_metadata_query,
    is_pg_dump_index_metadata_query, pg_dump_attrdef_metadata_columns,
    pg_dump_attrdef_metadata_query_relation_oids, pg_dump_attrdef_metadata_rows,
    pg_dump_attribute_metadata_columns, pg_dump_attribute_metadata_query_oids,
    pg_dump_attribute_metadata_rows, pg_dump_class_metadata_columns, pg_dump_class_metadata_rows,
    pg_dump_dependency_rows, pg_dump_index_metadata_columns, pg_dump_index_metadata_rows,
    pg_dump_table_oid_lookup_query_table, pg_dump_table_oid_lookup_rows,
    pg_dump_view_definition_query_oid, pg_dump_view_definition_rows,
};
use super::replication_catalog::try_execute_replication_pg_dump_query;
use super::role_ddl::try_execute_role_pg_dump_catalog_query;
use super::type_system_catalog::{
    try_execute_type_pg_dump_catalog_query, try_execute_type_system_pg_dump_catalog_query,
};
use super::{
    bool_column, catalog_empty_rows, function_acl_array_display, int4_column, int8_column,
    sql_type_display_name, text_column, write_error, write_single_row, CatalogCommentTarget,
    Column, ErrorField, ReadWrite, Session, PUBLIC_NAMESPACE_OID,
};
use std::io;

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

fn is_pg_dump_function_metadata_query(canonical: &str) -> bool {
    canonical
        .starts_with("select p.tableoid, p.oid, p.proname, p.prolang, p.pronargs, p.proargtypes")
        && canonical.contains("from pg_proc p")
}

pub(super) fn is_pg_dump_function_dump_prepare(canonical: &str) -> bool {
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

pub(super) fn pg_dump_function_dump_columns() -> Vec<Column> {
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

pub(super) fn pg_dump_function_dump_rows(
    session: &Session,
    oid: Option<u32>,
) -> Vec<Vec<Option<String>>> {
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

pub(super) fn pg_dump_empty_catalog_query_columns(canonical: &str) -> Option<Vec<Column>> {
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

pub(super) fn is_pg_dump_domain_constraints_prepare(canonical: &str) -> bool {
    canonical
        == "prepare getdomainconstraints(pg_catalog.oid) as select tableoid, oid, conname, pg_catalog.pg_get_constraintdef(oid) as consrc, convalidated from pg_catalog.pg_constraint where contypid = $1 order by conname"
}

pub(super) fn is_pg_dump_domain_constraints_execute(canonical: &str) -> bool {
    canonical.starts_with("execute getdomainconstraints(") && canonical.ends_with(')')
}

pub(super) fn pg_dump_domain_constraints_columns() -> Vec<Column> {
    vec![
        int4_column("tableoid"),
        int4_column("oid"),
        text_column("conname"),
        text_column("consrc"),
        bool_column("convalidated"),
    ]
}

pub(super) fn is_pg_dump_domain_dump_prepare(canonical: &str) -> bool {
    canonical
        == "prepare dumpdomain(pg_catalog.oid) as select t.typnotnull, pg_catalog.format_type(t.typbasetype, t.typtypmod) as typdefn, pg_catalog.pg_get_expr(t.typdefaultbin, 'pg_catalog.pg_type'::pg_catalog.regclass) as typdefaultbin, t.typdefault, case when t.typcollation <> u.typcollation then t.typcollation else 0 end as typcollation from pg_catalog.pg_type t left join pg_catalog.pg_type u on (t.typbasetype = u.oid) where t.oid = $1"
}

pub(super) fn pg_dump_domain_dump_execute_oid(canonical: &str) -> Option<u32> {
    canonical
        .strip_prefix("execute dumpdomain(")?
        .strip_suffix(')')?
        .trim_matches('\'')
        .parse()
        .ok()
}

pub(super) fn pg_dump_domain_dump_columns() -> Vec<Column> {
    vec![
        bool_column("typnotnull"),
        text_column("typdefn"),
        text_column("typdefaultbin"),
        text_column("typdefault"),
        int4_column("typcollation"),
    ]
}

pub(super) fn pg_dump_domain_dump_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
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
    if let Some(result) = try_execute_type_pg_dump_catalog_query(stream, session, canonical) {
        return Some(result);
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
    if let Some(result) = try_execute_access_method_pg_dump_catalog_query(stream, canonical) {
        return Some(result);
    }
    if let Some(result) = try_execute_aggregate_pg_dump_catalog_query(stream, canonical) {
        return Some(result);
    }
    if let Some(result) = try_execute_type_system_pg_dump_catalog_query(stream, canonical) {
        return Some(result);
    }
    if let Some(columns) = pg_dump_empty_catalog_query_columns(canonical) {
        return Some(write_single_row(stream, &columns, &catalog_empty_rows()));
    }
    if let Some(result) = try_execute_default_acl_pg_dump_catalog_query(stream, session, canonical)
    {
        return Some(result);
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
