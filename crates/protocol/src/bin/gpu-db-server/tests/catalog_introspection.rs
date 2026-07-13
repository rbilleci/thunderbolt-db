use super::*;

#[test]
fn catalog_helpers_expose_session_tables_and_columns() {
    let mut session = Session::default();
    session.tables.insert(
        "people".to_string(),
        Table {
            oid: FIRST_USER_RELATION_OID,
            name: "people".to_string(),
            columns: vec![
                CatalogColumn {
                    attnum: 1,
                    def: gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: gpu_db_protocol::SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: gpu_db_protocol::SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.tables.insert(
        "teams".to_string(),
        Table {
            oid: FIRST_USER_RELATION_OID + 1,
            name: "teams".to_string(),
            columns: vec![CatalogColumn {
                attnum: 1,
                def: gpu_db_protocol::ColumnDef {
                    name: "id".to_string(),
                    ty: gpu_db_protocol::SqlType::Int4,
                    domain: None,
                    default: None,
                },
            }],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );

    assert_eq!(
        catalog_table_name_rows(&session),
        vec![
            vec![Some("people".to_string())],
            vec![Some("teams".to_string())],
        ]
    );
    assert_eq!(
        catalog_table_oid_rows(&session),
        vec![
            vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("people".to_string()),
            ],
            vec![
                Some((FIRST_USER_RELATION_OID + 1).to_string()),
                Some("teams".to_string()),
            ],
        ]
    );
    assert_eq!(
        catalog_psql_describe_table_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("table".to_string()),
                Some("postgres".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("table".to_string()),
                Some("postgres".to_string()),
            ],
        ]
    );
    assert_eq!(
        psql_describe_tables_verbose_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        psql_describe_all_schema_tables_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') order by 1,2"
    );
    assert_eq!(
        psql_describe_all_schema_tables_verbose_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') order by 1,2"
    );
    assert_eq!(
        psql_describe_relations_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','v','m','s','f','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        catalog_psql_describe_table_verbose_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("table".to_string()),
                Some("postgres".to_string()),
                Some("permanent".to_string()),
                Some("heap".to_string()),
                Some("0 bytes".to_string()),
                None,
            ],
            vec![
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("table".to_string()),
                Some("postgres".to_string()),
                Some("permanent".to_string()),
                Some("heap".to_string()),
                Some("0 bytes".to_string()),
                None,
            ],
        ]
    );
    assert_eq!(
        psql_describe_tables_catalog_query_filter(
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1,2"
        ),
        Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: None,
        })
    );
    assert_eq!(
        psql_describe_tables_catalog_query_filter(
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and c.relname operator(pg_catalog.~) '^(people_.*)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1,2"
        ),
        Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some("people_.*".to_string()),
        })
    );
    assert_eq!(
        psql_describe_tables_catalog_query_filter(
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and c.relname operator(pg_catalog.~) '^(people_.*)$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        ),
        Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some("people_.*".to_string()),
        })
    );
    assert_eq!(
        psql_describe_tables_verbose_catalog_query_filter(
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        ),
        Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some("people".to_string()),
        })
    );
    assert_eq!(
        catalog_psql_describe_table_rows_filtered(
            &session,
            &PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: Some("peo.*".to_string()),
            },
        ),
        vec![vec![
            Some("public".to_string()),
            Some("people".to_string()),
            Some("table".to_string()),
            Some("postgres".to_string()),
        ]]
    );
    assert_eq!(
        catalog_psql_describe_table_verbose_rows_filtered(
            &session,
            &PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: Some("peo.*".to_string()),
            },
        ),
        vec![vec![
            Some("public".to_string()),
            Some("people".to_string()),
            Some("table".to_string()),
            Some("postgres".to_string()),
            Some("permanent".to_string()),
            Some("heap".to_string()),
            Some("0 bytes".to_string()),
            None,
        ]]
    );
    let mut sized_session = Session::default();
    sized_session.tables.insert(
        "people".to_string(),
        Table {
            oid: FIRST_USER_RELATION_OID,
            name: "people".to_string(),
            columns: vec![
                CatalogColumn {
                    attnum: 1,
                    def: gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: gpu_db_protocol::SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: gpu_db_protocol::SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("grace".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    assert_eq!(
        catalog_psql_describe_table_verbose_rows(&sized_session)[0][6],
        Some("64 bytes".to_string())
    );
    assert_eq!(
        psql_describe_tables_verbose_catalog_query_filter(
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and c.relname operator(pg_catalog.~) '^(peo.*)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1,2"
        ),
        Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some("peo.*".to_string()),
        })
    );
    assert_eq!(
        psql_describe_table_privileges_catalog_query_filter(
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 's' then 'sequence' when 'f' then 'foreign table' when 'p' then 'partitioned table' end as \"type\", pg_catalog.array_to_string(c.relacl, e'\\n') as \"access privileges\", pg_catalog.array_to_string(array( select attname || e':\\n ' || pg_catalog.array_to_string(attacl, e'\\n ') from pg_catalog.pg_attribute a where attrelid = c.oid and not attisdropped and attacl is not null ), e'\\n') as \"column privileges\", pg_catalog.array_to_string(array( select polname || case when not polpermissive then e' (restrictive)' else '' end || case when polcmd != '*' then e' (' || polcmd::pg_catalog.text || e'):' else e':' end || case when polqual is not null then e'\\n (u): ' || pg_catalog.pg_get_expr(polqual, polrelid) else e'' end || case when polwithcheck is not null then e'\\n (c): ' || pg_catalog.pg_get_expr(polwithcheck, polrelid) else e'' end || case when polroles <> '{0}' then e'\\n to: ' || pg_catalog.array_to_string( array( select rolname from pg_catalog.pg_roles where oid = any (polroles) order by 1 ), e', ') else e'' end from pg_catalog.pg_policy pol where polrelid = c.oid), e'\\n') as \"policies\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('r','v','m','s','f','p') and c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1, 2"
        ),
        Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some("people".to_string()),
        })
    );
    assert_eq!(
        catalog_psql_describe_table_privilege_rows_filtered(
            &session,
            &PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: Some("peo.*".to_string()),
            },
        ),
        vec![vec![
            Some("public".to_string()),
            Some("people".to_string()),
            Some("table".to_string()),
            None,
            None,
            None,
        ]]
    );
    assert_eq!(
        psql_describe_indexes_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        psql_describe_indexes_catalog_query_schema_filter(
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','s','') and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1,2"
        ),
        Some("public".to_string())
    );
    assert_eq!(
        psql_describe_indexes_catalog_query_schema_filter(
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','s','') and n.nspname operator(pg_catalog.~) '^(private)$' collate pg_catalog.default order by 1,2"
        ),
        None
    );
    assert_eq!(
        psql_describe_views_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('v','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        psql_describe_views_verbose_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('v','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        psql_describe_materialized_views_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('m','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        psql_describe_materialized_views_verbose_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('m','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        psql_describe_sequences_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('s','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        psql_describe_sequences_verbose_catalog_query(),
        "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('s','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
    );
    assert_eq!(
        psql_describe_functions_catalog_query(),
        "select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.pg_get_function_result(p.oid) as \"result data type\", pg_catalog.pg_get_function_arguments(p.oid) as \"argument data types\", case p.prokind when 'a' then 'agg' when 'w' then 'window' when 'p' then 'proc' else 'func' end as \"type\" from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n on n.oid = p.pronamespace where pg_catalog.pg_function_is_visible(p.oid) and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' order by 1, 2, 4"
    );
    assert_eq!(
        psql_list_extensions_catalog_query(),
        "select e.extname as \"name\", e.extversion as \"version\", n.nspname as \"schema\", c.description as \"description\" from pg_catalog.pg_extension e left join pg_catalog.pg_namespace n on n.oid = e.extnamespace left join pg_catalog.pg_description c on c.objoid = e.oid and c.classoid = 'pg_catalog.pg_extension'::pg_catalog.regclass order by 1"
    );
    assert_eq!(
        psql_list_languages_catalog_query(),
        "select l.lanname as \"name\", pg_catalog.pg_get_userbyid(l.lanowner) as \"owner\", l.lanpltrusted as \"trusted\", d.description as \"description\" from pg_catalog.pg_language l left join pg_catalog.pg_description d on d.classoid = l.tableoid and d.objoid = l.oid and d.objsubid = 0 where l.lanplcallfoid != 0 order by 1"
    );
    assert_eq!(
        psql_describe_roles_catalog_query(),
        "select r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, r.rolcreatedb, r.rolcanlogin, r.rolconnlimit, r.rolvaliduntil , r.rolreplication , r.rolbypassrls from pg_catalog.pg_roles r where r.rolname !~ '^pg_' order by 1"
    );
    assert_eq!(
        psql_describe_roles_verbose_catalog_query(),
        "select r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, r.rolcreatedb, r.rolcanlogin, r.rolconnlimit, r.rolvaliduntil , pg_catalog.shobj_description(r.oid, 'pg_authid') as description , r.rolreplication , r.rolbypassrls from pg_catalog.pg_roles r where r.rolname !~ '^pg_' order by 1"
    );
    assert_eq!(
        catalog_psql_describe_role_rows(&Session::default(), false),
        vec![vec![
            Some("postgres".to_string()),
            Some("t".to_string()),
            Some("t".to_string()),
            Some("t".to_string()),
            Some("t".to_string()),
            Some("t".to_string()),
            Some("-1".to_string()),
            None,
            Some("t".to_string()),
            Some("t".to_string()),
        ]]
    );
    let mut commented_role = Session::default();
    commented_role.comments.insert(
        CatalogCommentTarget::Role {
            role: "postgres".to_string(),
        },
        "bootstrap role".to_string(),
    );
    assert_eq!(
        catalog_psql_describe_role_rows(&commented_role, true)[0][8],
        Some("bootstrap role".to_string())
    );
    assert_eq!(
        psql_list_databases_catalog_query(),
        "select d.datname as \"name\", pg_catalog.pg_get_userbyid(d.datdba) as \"owner\", pg_catalog.pg_encoding_to_char(d.encoding) as \"encoding\", case d.datlocprovider when 'c' then 'libc' when 'i' then 'icu' end as \"locale provider\", d.datcollate as \"collate\", d.datctype as \"ctype\", d.daticulocale as \"icu locale\", d.daticurules as \"icu rules\", pg_catalog.array_to_string(d.datacl, e'\\n') as \"access privileges\" from pg_catalog.pg_database d order by 1"
    );
    assert_eq!(
        psql_list_databases_verbose_catalog_query(),
        "select d.datname as \"name\", pg_catalog.pg_get_userbyid(d.datdba) as \"owner\", pg_catalog.pg_encoding_to_char(d.encoding) as \"encoding\", case d.datlocprovider when 'c' then 'libc' when 'i' then 'icu' end as \"locale provider\", d.datcollate as \"collate\", d.datctype as \"ctype\", d.daticulocale as \"icu locale\", d.daticurules as \"icu rules\", pg_catalog.array_to_string(d.datacl, e'\\n') as \"access privileges\", case when pg_catalog.has_database_privilege(d.datname, 'connect') then pg_catalog.pg_size_pretty(pg_catalog.pg_database_size(d.datname)) else 'no access' end as \"size\", t.spcname as \"tablespace\", pg_catalog.shobj_description(d.oid, 'pg_database') as \"description\" from pg_catalog.pg_database d join pg_catalog.pg_tablespace t on d.dattablespace = t.oid order by 1"
    );
    assert_eq!(
        catalog_psql_list_database_rows(&Session::default()),
        vec![vec![
            Some("postgres".to_string()),
            Some("postgres".to_string()),
            Some("UTF8".to_string()),
            Some("libc".to_string()),
            Some("C.UTF-8".to_string()),
            Some("C.UTF-8".to_string()),
            None,
            None,
            None,
        ]]
    );
    let mut commented_database = Session::default();
    commented_database.comments.insert(
        CatalogCommentTarget::Database {
            database: "postgres".to_string(),
        },
        "primary database".to_string(),
    );
    commented_database.databases.insert(
        "appdb".to_string(),
        DatabaseInfo {
            oid: FIRST_USER_RELATION_OID,
            name: "appdb".to_string(),
        },
    );
    commented_database.comments.insert(
        CatalogCommentTarget::Database {
            database: "appdb".to_string(),
        },
        "application database".to_string(),
    );
    commented_database
        .database_acls
        .entry("appdb".to_string())
        .or_default()
        .insert(
            "app_reader".to_string(),
            BTreeSet::from([DatabasePrivilege::Connect, DatabasePrivilege::Temporary]),
        );
    assert_eq!(
        catalog_psql_list_database_verbose_rows(&commented_database)[0][11],
        Some("application database".to_string())
    );
    assert_eq!(
        catalog_psql_list_database_verbose_rows(&commented_database)[0][8],
        Some("app_reader=cT/postgres".to_string())
    );
    assert_eq!(
        catalog_database_acl_rows(&commented_database)[0],
        vec![
            Some("appdb".to_string()),
            Some("app_reader=cT/postgres".to_string())
        ]
    );
    assert_eq!(
        catalog_psql_list_database_verbose_rows(&commented_database)[1][11],
        Some("primary database".to_string())
    );
    assert_eq!(
        catalog_psql_list_database_rows(&commented_database)
            .into_iter()
            .map(|row| row[0].clone())
            .collect::<Vec<_>>(),
        vec![Some("appdb".to_string()), Some("postgres".to_string())]
    );
    assert_eq!(
        catalog_database_oid_rows(&commented_database),
        vec![
            vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("appdb".to_string())
            ],
            vec![
                Some(POSTGRES_DATABASE_OID.to_string()),
                Some("postgres".to_string())
            ],
        ]
    );
    assert_eq!(
        psql_list_tablespaces_catalog_query(),
        "select spcname as \"name\", pg_catalog.pg_get_userbyid(spcowner) as \"owner\", pg_catalog.pg_tablespace_location(oid) as \"location\" from pg_catalog.pg_tablespace order by 1"
    );
    assert_eq!(
        psql_list_tablespaces_verbose_catalog_query(),
        "select spcname as \"name\", pg_catalog.pg_get_userbyid(spcowner) as \"owner\", pg_catalog.pg_tablespace_location(oid) as \"location\", pg_catalog.array_to_string(spcacl, e'\\n') as \"access privileges\", spcoptions as \"options\", pg_catalog.pg_size_pretty(pg_catalog.pg_tablespace_size(oid)) as \"size\", pg_catalog.shobj_description(oid, 'pg_tablespace') as \"description\" from pg_catalog.pg_tablespace order by 1"
    );
    assert_eq!(
        catalog_psql_list_tablespace_rows(&Session::default(), false),
        vec![
            vec![
                Some("pg_default".to_string()),
                Some("postgres".to_string()),
                Some(String::new()),
            ],
            vec![
                Some("pg_global".to_string()),
                Some("postgres".to_string()),
                Some(String::new()),
            ],
        ]
    );
    let mut commented_tablespace = Session::default();
    commented_tablespace.comments.insert(
        CatalogCommentTarget::Tablespace {
            tablespace: "pg_default".to_string(),
        },
        "default storage".to_string(),
    );
    commented_tablespace
        .tablespace_acls
        .entry("pg_default".to_string())
        .or_default()
        .insert(
            "app_reader".to_string(),
            BTreeSet::from([TablespacePrivilege::Create]),
        );
    assert_eq!(
        catalog_psql_list_tablespace_rows(&commented_tablespace, true)[0][6],
        Some("default storage".to_string())
    );
    assert_eq!(
        catalog_psql_list_tablespace_rows(&commented_tablespace, true)[0][3],
        Some("app_reader=C/postgres".to_string())
    );
    assert_eq!(
        catalog_tablespace_acl_rows(&commented_tablespace)[0],
        vec![
            Some("pg_default".to_string()),
            Some("app_reader=C/postgres".to_string())
        ]
    );
    assert_eq!(
        psql_list_access_methods_catalog_query(),
        "select amname as \"name\", case amtype when 'i' then 'index' when 't' then 'table' end as \"type\" from pg_catalog.pg_am order by 1"
    );
    assert_eq!(
        catalog_psql_list_access_method_rows(),
        vec![vec![Some("heap".to_string()), Some("Table".to_string())]]
    );
    assert!(catalog_empty_rows().is_empty());
    assert!(catalog_psql_describe_table_rows_filtered(
        &session,
        &PsqlDescribeTablesFilter {
            namespace: "private".to_string(),
            relname_pattern: None,
        },
    )
    .is_empty());
    assert!(psql_relname_pattern_matches("peo.*", "people"));
    assert!(!psql_relname_pattern_matches("tea.*", "people"));
    assert!(psql_relname_pattern_matches("people", "people"));
    assert!(!psql_relname_pattern_matches("people", "teams"));
    assert_eq!(
        psql_describe_schemas_catalog_query(),
        "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\" from pg_catalog.pg_namespace n where n.nspname !~ '^pg_' and n.nspname <> 'information_schema' order by 1"
    );
    assert!(psql_describe_schemas_verbose_catalog_query_public_filter(
        "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\", pg_catalog.array_to_string(n.nspacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(n.oid, 'pg_namespace') as \"description\" from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1"
    ));
    assert!(!psql_describe_schemas_verbose_catalog_query_public_filter(
        "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\", pg_catalog.array_to_string(n.nspacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(n.oid, 'pg_namespace') as \"description\" from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(private)$' collate pg_catalog.default order by 1"
    ));
    assert_eq!(
        psql_describe_schema_publications_query(),
        "select pubname from pg_catalog.pg_publication p join pg_catalog.pg_publication_namespace pn on p.oid = pn.pnpubid join pg_catalog.pg_namespace n on n.oid = pn.pnnspid where n.nspname = 'public' order by 1"
    );
    assert_eq!(
        psql_list_domains_catalog_query(),
        "select n.nspname as \"schema\", t.typname as \"name\", pg_catalog.format_type(t.typbasetype, t.typtypmod) as \"type\", (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type bt where c.oid = t.typcollation and bt.oid = t.typbasetype and t.typcollation <> bt.typcollation) as \"collation\", case when t.typnotnull then 'not null' end as \"nullable\", t.typdefault as \"default\", pg_catalog.array_to_string(array( select pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where t.oid = r.contypid ), ' ') as \"check\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where t.typtype = 'd' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_type_is_visible(t.oid) order by 1, 2"
    );
    assert_eq!(
        psql_list_domains_verbose_catalog_query(),
        "select n.nspname as \"schema\", t.typname as \"name\", pg_catalog.format_type(t.typbasetype, t.typtypmod) as \"type\", (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type bt where c.oid = t.typcollation and bt.oid = t.typbasetype and t.typcollation <> bt.typcollation) as \"collation\", case when t.typnotnull then 'not null' end as \"nullable\", t.typdefault as \"default\", pg_catalog.array_to_string(array( select pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where t.oid = r.contypid ), ' ') as \"check\", pg_catalog.array_to_string(t.typacl, e'\\n') as \"access privileges\", d.description as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace left join pg_catalog.pg_description d on d.classoid = t.tableoid and d.objoid = t.oid and d.objsubid = 0 where t.typtype = 'd' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_type_is_visible(t.oid) order by 1, 2"
    );
    assert_eq!(
        psql_list_aggregates_catalog_query(),
        "select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.format_type(p.prorettype, null) as \"result data type\", case when p.pronargs = 0 then cast('*' as pg_catalog.text) else pg_catalog.pg_get_function_arguments(p.oid) end as \"argument data types\", pg_catalog.obj_description(p.oid, 'pg_proc') as \"description\" from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n on n.oid = p.pronamespace where p.prokind = 'a' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_function_is_visible(p.oid) order by 1, 2, 4"
    );
    assert_eq!(
        psql_list_conversions_catalog_query(),
        "select n.nspname as \"schema\", c.conname as \"name\", pg_catalog.pg_encoding_to_char(c.conforencoding) as \"source\", pg_catalog.pg_encoding_to_char(c.contoencoding) as \"destination\", case when c.condefault then 'yes' else 'no' end as \"default?\" from pg_catalog.pg_conversion c join pg_catalog.pg_namespace n on n.oid = c.connamespace where true and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_conversion_is_visible(c.oid) order by 1, 2"
    );
    assert_eq!(
        psql_list_operators_catalog_query(),
        "select n.nspname as \"schema\", o.oprname as \"name\", case when o.oprkind='l' then null else pg_catalog.format_type(o.oprleft, null) end as \"left arg type\", case when o.oprkind='r' then null else pg_catalog.format_type(o.oprright, null) end as \"right arg type\", pg_catalog.format_type(o.oprresult, null) as \"result type\", coalesce(pg_catalog.obj_description(o.oid, 'pg_operator'), pg_catalog.obj_description(o.oprcode, 'pg_proc')) as \"description\" from pg_catalog.pg_operator o left join pg_catalog.pg_namespace n on n.oid = o.oprnamespace where n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_operator_is_visible(o.oid) order by 1, 2, 3, 4"
    );
    assert_eq!(
        psql_list_collations_catalog_query(),
        "select n.nspname as \"schema\", c.collname as \"name\", case c.collprovider when 'd' then 'default' when 'c' then 'libc' when 'i' then 'icu' end as \"provider\", c.collcollate as \"collate\", c.collctype as \"ctype\", c.colliculocale as \"icu locale\", c.collicurules as \"icu rules\", case when c.collisdeterministic then 'yes' else 'no' end as \"deterministic?\" from pg_catalog.pg_collation c, pg_catalog.pg_namespace n where n.oid = c.collnamespace and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and c.collencoding in (-1, pg_catalog.pg_char_to_encoding(pg_catalog.getdatabaseencoding())) and pg_catalog.pg_collation_is_visible(c.oid) order by 1, 2"
    );
    assert_eq!(
        psql_list_casts_catalog_query(),
        "select pg_catalog.format_type(castsource, null) as \"source type\", pg_catalog.format_type(casttarget, null) as \"target type\", case when c.castmethod = 'b' then '(binary coercible)' when c.castmethod = 'i' then '(with inout)' else p.proname end as \"function\", case when c.castcontext = 'e' then 'no' when c.castcontext = 'a' then 'in assignment' else 'yes' end as \"implicit?\" from pg_catalog.pg_cast c left join pg_catalog.pg_proc p on c.castfunc = p.oid left join pg_catalog.pg_type ts on c.castsource = ts.oid left join pg_catalog.pg_namespace ns on ns.oid = ts.typnamespace left join pg_catalog.pg_type tt on c.casttarget = tt.oid left join pg_catalog.pg_namespace nt on nt.oid = tt.typnamespace where ( (true and pg_catalog.pg_type_is_visible(ts.oid) ) or (true and pg_catalog.pg_type_is_visible(tt.oid) ) ) order by 1, 2"
    );
    assert_eq!(
        psql_list_publications_catalog_query(),
        "select pubname as \"name\", pg_catalog.pg_get_userbyid(pubowner) as \"owner\", puballtables as \"all tables\", pubinsert as \"inserts\", pubupdate as \"updates\", pubdelete as \"deletes\", pubtruncate as \"truncates\", pubviaroot as \"via root\" from pg_catalog.pg_publication order by 1"
    );
    assert_eq!(
        psql_list_publications_verbose_catalog_query(),
        "select oid, pubname, pg_catalog.pg_get_userbyid(pubowner) as owner, puballtables, pubinsert, pubupdate, pubdelete, pubtruncate, pubviaroot from pg_catalog.pg_publication order by 2"
    );
    assert_eq!(
        psql_list_subscriptions_catalog_query(),
        "select subname as \"name\" , pg_catalog.pg_get_userbyid(subowner) as \"owner\" , subenabled as \"enabled\" , subpublications as \"publication\" from pg_catalog.pg_subscription where subdbid = (select oid from pg_catalog.pg_database where datname = pg_catalog.current_database())order by 1"
    );
    assert_eq!(
        psql_list_default_access_privileges_catalog_query(),
        "select pg_catalog.pg_get_userbyid(d.defaclrole) as \"owner\", n.nspname as \"schema\", case d.defaclobjtype when 'r' then 'table' when 's' then 'sequence' when 'f' then 'function' when 't' then 'type' when 'n' then 'schema' end as \"type\", pg_catalog.array_to_string(d.defaclacl, e'\\n') as \"access privileges\" from pg_catalog.pg_default_acl d left join pg_catalog.pg_namespace n on n.oid = d.defaclnamespace order by 1, 2, 3"
    );
    assert_eq!(
        catalog_psql_describe_schema_rows(&Session::default()),
        vec![vec![
            Some("public".to_string()),
            Some("postgres".to_string())
        ]]
    );
    let default_schema_session = Session::default();
    assert_eq!(
        catalog_psql_describe_schema_verbose_rows(&default_schema_session),
        vec![vec![
            Some("public".to_string()),
            Some("postgres".to_string()),
            None,
            None
        ]]
    );
    assert_eq!(
        psql_describe_type_catalog_query_type(
            "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and (t.typname operator(pg_catalog.~) '^(int4)$' collate pg_catalog.default or pg_catalog.format_type(t.oid, null) operator(pg_catalog.~) '^(int4)$' collate pg_catalog.default) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
        ),
        Some("int4".to_string())
    );
    assert_eq!(
        psql_describe_type_catalog_query_type(
            "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and (t.typname operator(pg_catalog.~) '^(text)$' collate pg_catalog.default or pg_catalog.format_type(t.oid, null) operator(pg_catalog.~) '^(text)$' collate pg_catalog.default) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
        ),
        Some("text".to_string())
    );
    assert_eq!(
        catalog_psql_describe_type_rows("int4"),
        vec![vec![
            Some("pg_catalog".to_string()),
            Some("integer".to_string()),
            None
        ]]
    );
    assert_eq!(
        catalog_psql_describe_type_rows("text"),
        vec![vec![
            Some("pg_catalog".to_string()),
            Some("text".to_string()),
            None
        ]]
    );
    assert_eq!(
        psql_describe_pg_catalog_types_query(),
        "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
    );
    assert_eq!(
        catalog_psql_describe_type_rows_for_supported_types(),
        vec![
            vec![
                Some("pg_catalog".to_string()),
                Some("bigint".to_string()),
                None
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("boolean".to_string()),
                None
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("date".to_string()),
                None
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("integer".to_string()),
                None
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("numeric".to_string()),
                None
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("smallint".to_string()),
                None
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("text".to_string()),
                None
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("timestamp without time zone".to_string()),
                None
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("uuid".to_string()),
                None
            ],
        ]
    );
    assert_eq!(
        catalog_psql_describe_type_verbose_rows_for_supported_types(),
        vec![
            vec![
                Some("pg_catalog".to_string()),
                Some("bigint".to_string()),
                Some("int8".to_string()),
                Some(String::new()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("boolean".to_string()),
                Some("bool".to_string()),
                Some(String::new()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("date".to_string()),
                Some("date".to_string()),
                Some("4".to_string()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("integer".to_string()),
                Some("int4".to_string()),
                Some("4".to_string()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("numeric".to_string()),
                Some("numeric".to_string()),
                Some("var".to_string()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("smallint".to_string()),
                Some("int2".to_string()),
                Some(String::new()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("text".to_string()),
                Some("text".to_string()),
                Some("var".to_string()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("timestamp without time zone".to_string()),
                Some("timestamp".to_string()),
                Some(String::new()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("pg_catalog".to_string()),
                Some("uuid".to_string()),
                Some("uuid".to_string()),
                Some(String::new()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ],
        ]
    );
    assert_eq!(
        catalog_describe_relation_lookup_query_table(
            "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 2, 3"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        catalog_describe_relation_lookup_query_table(
            "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        catalog_describe_relation_lookup_query_table(
            "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(peo.*)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
        ),
        Some("peo.*".to_string())
    );
    assert_eq!(
        catalog_describe_relation_lookup_query_table(
            "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(private)$' collate pg_catalog.default order by 2, 3"
        ),
        None
    );
    assert!(catalog_describe_relation_lookup_query_public_namespace(
        "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
    ));
    assert!(catalog_describe_relation_lookup_query_all_schemas(
        "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace order by 2, 3"
    ));
    assert_eq!(
        catalog_describe_relation_lookup_rows(&session, "people"),
        vec![vec![
            Some(FIRST_USER_RELATION_OID.to_string()),
            Some("public".to_string()),
            Some("people".to_string()),
        ]]
    );
    assert_eq!(
        catalog_describe_relation_lookup_rows(&session, "peo.*"),
        vec![vec![
            Some(FIRST_USER_RELATION_OID.to_string()),
            Some("public".to_string()),
            Some("people".to_string()),
        ]]
    );
    assert!(catalog_describe_relation_lookup_rows(&session, "missing").is_empty());
    assert_eq!(
        catalog_describe_relation_lookup_rows_for_public_namespace(&session),
        vec![
            vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
            ],
            vec![
                Some((FIRST_USER_RELATION_OID + 1).to_string()),
                Some("public".to_string()),
                Some("teams".to_string()),
            ],
        ]
    );
    assert_eq!(
        catalog_describe_relation_flags_query_oid(
            "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, '', c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '16384'"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        catalog_describe_relation_flags_query_oid(
            "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, pg_catalog.array_to_string(c.reloptions || array(select 'toast.' || x from pg_catalog.unnest(tc.reloptions) x), ', ') , c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '16384'"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        catalog_describe_relation_flags_rows(&session, FIRST_USER_RELATION_OID),
        vec![vec![
            Some("0".to_string()),
            Some("r".to_string()),
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
        ]]
    );
    assert_eq!(
        catalog_describe_attribute_query_oid(
            "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated from pg_catalog.pg_attribute a where a.attrelid = '16384' and a.attnum > 0 and not a.attisdropped order by a.attnum"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        catalog_describe_attribute_rows(&session, FIRST_USER_RELATION_OID),
        vec![
            vec![
                Some("id".to_string()),
                Some("integer".to_string()),
                None,
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
            ],
            vec![
                Some("name".to_string()),
                Some("text".to_string()),
                None,
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
            ],
        ]
    );
    assert_eq!(
        catalog_describe_verbose_attribute_query_oid(
            "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated, a.attstorage, a.attcompression as attcompression, case when a.attstattarget=-1 then null else a.attstattarget end as attstattarget, pg_catalog.col_description(a.attrelid, a.attnum) from pg_catalog.pg_attribute a where a.attrelid = '16384' and a.attnum > 0 and not a.attisdropped order by a.attnum"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        catalog_describe_verbose_attribute_rows(&session, FIRST_USER_RELATION_OID),
        vec![
            vec![
                Some("id".to_string()),
                Some("integer".to_string()),
                None,
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
                Some("p".to_string()),
                Some(String::new()),
                None,
                None,
            ],
            vec![
                Some("name".to_string()),
                Some("text".to_string()),
                None,
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
                Some("x".to_string()),
                Some(String::new()),
                None,
                None,
            ],
        ]
    );
    assert_eq!(
        catalog_describe_policy_query_oid(
            "select pol.polname, pol.polpermissive, case when pol.polroles = '{0}' then null else pg_catalog.array_to_string(array(select rolname from pg_catalog.pg_roles where oid = any (pol.polroles) order by 1),',') end, pg_catalog.pg_get_expr(pol.polqual, pol.polrelid), pg_catalog.pg_get_expr(pol.polwithcheck, pol.polrelid), case pol.polcmd when 'r' then 'select' when 'a' then 'insert' when 'w' then 'update' when 'd' then 'delete' end as cmd from pg_catalog.pg_policy pol where pol.polrelid = '16384' order by 1"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        catalog_describe_statistic_ext_query_oid(
            "select oid, stxrelid::pg_catalog.regclass, stxnamespace::pg_catalog.regnamespace::pg_catalog.text as nsp, stxname, pg_catalog.pg_get_statisticsobjdef_columns(oid) as columns, 'd' = any(stxkind) as ndist_enabled, 'f' = any(stxkind) as deps_enabled, 'm' = any(stxkind) as mcv_enabled, stxstattarget from pg_catalog.pg_statistic_ext where stxrelid = '16384' order by nsp, stxname"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        catalog_describe_publication_query_oid(
            "select pubname , null , null from pg_catalog.pg_publication p join pg_catalog.pg_publication_namespace pn on p.oid = pn.pnpubid join pg_catalog.pg_class pc on pc.relnamespace = pn.pnnspid where pc.oid ='16384' and pg_catalog.pg_relation_is_publishable('16384') union select pubname , pg_get_expr(pr.prqual, c.oid) , (case when pr.prattrs is not null then (select string_agg(attname, ', ') from pg_catalog.generate_series(0, pg_catalog.array_upper(pr.prattrs::pg_catalog.int2[], 1)) s, pg_catalog.pg_attribute where attrelid = pr.prrelid and attnum = prattrs[s]) else null end) from pg_catalog.pg_publication p join pg_catalog.pg_publication_rel pr on p.oid = pr.prpubid join pg_catalog.pg_class c on c.oid = pr.prrelid where pr.prrelid = '16384' union select pubname , null , null from pg_catalog.pg_publication p where p.puballtables and pg_catalog.pg_relation_is_publishable('16384') order by 1"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        catalog_describe_inherits_parent_query_oid(
            "select c.oid::pg_catalog.regclass from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhparent and i.inhrelid = '16384' and c.relkind != 'p' and c.relkind != 'i' order by inhseqno"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        catalog_describe_inherits_child_query_oid(
            "select c.oid::pg_catalog.regclass, c.relkind, inhdetachpending, pg_catalog.pg_get_expr(c.relpartbound, c.oid) from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhrelid and i.inhparent = '16384' order by pg_catalog.pg_get_expr(c.relpartbound, c.oid) = 'default', c.oid::pg_catalog.regclass::pg_catalog.text"
        ),
        Some(FIRST_USER_RELATION_OID)
    );
    assert_eq!(
        pg_catalog_tables_query(),
        "select schemaname, tablename, tableowner from pg_catalog.pg_tables where schemaname = 'public' order by tablename"
    );
    assert_eq!(
        pg_catalog_table_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("postgres".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("postgres".to_string()),
            ],
        ]
    );
    assert_eq!(
        pg_catalog_indexes_query(),
        "select schemaname, tablename, indexname, indexdef from pg_catalog.pg_indexes where schemaname = 'public' order by tablename, indexname"
    );
    assert!(pg_catalog_index_rows(&session).is_empty());
    assert_eq!(
        pg_catalog_class_plain_tables_query(),
        "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relkind = 'r' order by c.relname"
    );
    assert_eq!(
        pg_catalog_class_plain_table_rows(&session),
        vec![
            vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("r".to_string()),
                Some("p".to_string()),
            ],
            vec![
                Some((FIRST_USER_RELATION_OID + 1).to_string()),
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("r".to_string()),
                Some("p".to_string()),
            ],
        ]
    );
    assert_eq!(
        pg_catalog_class_plain_tables_in_query_tables(
            "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname in ('teams', 'missing', 'people') and c.relkind = 'r' order by c.relname"
        ),
        Some(vec![
            "teams".to_string(),
            "missing".to_string(),
            "people".to_string(),
        ])
    );
    assert_eq!(
        pg_catalog_class_plain_table_rows_for_tables(
            &session,
            &[
                "teams".to_string(),
                "missing".to_string(),
                "people".to_string(),
                "teams".to_string(),
            ],
        ),
        vec![
            vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("r".to_string()),
                Some("p".to_string()),
            ],
            vec![
                Some((FIRST_USER_RELATION_OID + 1).to_string()),
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("r".to_string()),
                Some("p".to_string()),
            ],
        ]
    );
    assert_eq!(
        pg_catalog_table_descriptions_query(),
        "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind = 'r' order by c.relname, d.objsubid"
    );
    assert!(pg_catalog_description_rows(&session).is_empty());
    assert!(psql_list_object_descriptions_query(&canonical_sql(
        "SELECT DISTINCT tt.nspname AS \"Schema\", tt.name AS \"Name\", tt.object AS \"Object\", d.description AS \"Description\"
         FROM (
           SELECT pgc.oid as oid, pgc.tableoid AS tableoid,
           n.nspname as nspname,
           CAST(pgc.conname AS pg_catalog.text) as name, CAST('table constraint' AS pg_catalog.text) as object
           FROM pg_catalog.pg_constraint pgc
           JOIN pg_catalog.pg_class c ON c.oid = pgc.conrelid
           LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname <> 'pg_catalog' AND n.nspname <> 'information_schema'
             AND pg_catalog.pg_table_is_visible(c.oid)
         UNION ALL
           SELECT pgc.oid as oid, pgc.tableoid AS tableoid,
           n.nspname as nspname,
           CAST(pgc.conname AS pg_catalog.text) as name, CAST('domain constraint' AS pg_catalog.text) as object
           FROM pg_catalog.pg_constraint pgc
           JOIN pg_catalog.pg_type t ON t.oid = pgc.contypid
           LEFT JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
           WHERE n.nspname <> 'pg_catalog' AND n.nspname <> 'information_schema'
             AND pg_catalog.pg_type_is_visible(t.oid)
         UNION ALL
           SELECT o.oid as oid, o.tableoid as tableoid,
           n.nspname as nspname,
           CAST(o.opcname AS pg_catalog.text) as name,
           CAST('operator class' AS pg_catalog.text) as object
           FROM pg_catalog.pg_opclass o
           JOIN pg_catalog.pg_am am ON o.opcmethod = am.oid
           JOIN pg_catalog.pg_namespace n ON n.oid = o.opcnamespace
             AND n.nspname <> 'pg_catalog'
             AND n.nspname <> 'information_schema'
             AND pg_catalog.pg_opclass_is_visible(o.oid)
         UNION ALL
           SELECT opf.oid as oid, opf.tableoid as tableoid,
           n.nspname as nspname,
           CAST(opf.opfname AS pg_catalog.text) AS name,
           CAST('operator family' AS pg_catalog.text) as object
           FROM pg_catalog.pg_opfamily opf
           JOIN pg_catalog.pg_am am ON opf.opfmethod = am.oid
           JOIN pg_catalog.pg_namespace n ON opf.opfnamespace = n.oid
             AND n.nspname <> 'pg_catalog'
             AND n.nspname <> 'information_schema'
             AND pg_catalog.pg_opfamily_is_visible(opf.oid)
         UNION ALL
           SELECT r.oid as oid, r.tableoid as tableoid,
           n.nspname as nspname,
           CAST(r.rulename AS pg_catalog.text) as name, CAST('rule' AS pg_catalog.text) as object
           FROM pg_catalog.pg_rewrite r
           JOIN pg_catalog.pg_class c ON c.oid = r.ev_class
           LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
           WHERE r.rulename != '_RETURN'
             AND n.nspname <> 'pg_catalog'
             AND n.nspname <> 'information_schema'
             AND pg_catalog.pg_table_is_visible(c.oid)
         UNION ALL
           SELECT t.oid as oid, t.tableoid as tableoid,
           n.nspname as nspname,
           CAST(t.tgname AS pg_catalog.text) as name, CAST('trigger' AS pg_catalog.text) as object
           FROM pg_catalog.pg_trigger t
           JOIN pg_catalog.pg_class c ON c.oid = t.tgrelid
           LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname <> 'pg_catalog'
             AND n.nspname <> 'information_schema'
             AND pg_catalog.pg_table_is_visible(c.oid)
         ) AS tt
         JOIN pg_catalog.pg_description d ON (tt.oid = d.objoid AND tt.tableoid = d.classoid AND d.objsubid = 0)
         ORDER BY 1, 2, 3;"
    )));
    assert_eq!(
        information_schema_table_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("BASE TABLE".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("BASE TABLE".to_string()),
            ],
        ]
    );
    assert_eq!(
        information_schema_tables_in_query_tables(
            "select table_schema, table_name, table_type from information_schema.tables where table_schema = 'public' and table_name in ('people', 'missing', 'teams') order by table_name"
        ),
        Some(vec![
            "people".to_string(),
            "missing".to_string(),
            "teams".to_string()
        ])
    );
    assert_eq!(
        information_schema_table_rows_for_tables(
            &session,
            &[
                "people".to_string(),
                "missing".to_string(),
                "people".to_string(),
                "teams".to_string()
            ]
        ),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("BASE TABLE".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("BASE TABLE".to_string()),
            ],
        ]
    );
    assert_eq!(
        information_schema_base_table_discovery_query(),
        "select table_schema, table_name from information_schema.tables where table_type = 'base table' and table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name"
    );
    assert_eq!(
        information_schema_base_table_discovery_rows(&session),
        vec![
            vec![Some("public".to_string()), Some("people".to_string())],
            vec![Some("public".to_string()), Some("teams".to_string())],
        ]
    );
    assert_eq!(
        information_schema_rich_tables_query(),
        "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_schema = 'public' order by table_name"
    );
    assert_eq!(
        information_schema_rich_table_rows(&session),
        vec![
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("BASE TABLE".to_string()),
                None,
                None,
                None,
                None,
                None,
                Some("YES".to_string()),
                Some("NO".to_string()),
                None,
            ],
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("BASE TABLE".to_string()),
                None,
                None,
                None,
                None,
                None,
                Some("YES".to_string()),
                Some("NO".to_string()),
                None,
            ],
        ]
    );
    assert_eq!(
        information_schema_rich_tables_query_table(
            "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_schema = 'public' and table_name = 'people' order by table_name"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        information_schema_rich_table_rows_for_table(&session, "people"),
        vec![vec![
            Some("postgres".to_string()),
            Some("public".to_string()),
            Some("people".to_string()),
            Some("BASE TABLE".to_string()),
            None,
            None,
            None,
            None,
            None,
            Some("YES".to_string()),
            Some("NO".to_string()),
            None,
        ]]
    );
    assert!(information_schema_rich_table_rows_for_table(&session, "missing").is_empty());
    assert_eq!(
        information_schema_columns_query_table(
            "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name = 'people' order by ordinal_position"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        information_schema_column_rows(&session, "people"),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                Some("integer".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("name".to_string()),
                Some("2".to_string()),
                Some("text".to_string()),
            ],
        ]
    );
    assert_eq!(
        information_schema_all_columns_query(),
        "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
    );
    assert_eq!(
        information_schema_column_discovery_query(),
        "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name, ordinal_position"
    );
    assert_eq!(
        information_schema_all_column_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                Some("integer".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("name".to_string()),
                Some("2".to_string()),
                Some("text".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                Some("integer".to_string()),
            ],
        ]
    );
    assert_eq!(
        information_schema_columns_in_query_tables(
            "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name in ('people', 'missing', 'teams') order by table_name, ordinal_position"
        ),
        Some(vec![
            "people".to_string(),
            "missing".to_string(),
            "teams".to_string()
        ])
    );
    assert_eq!(
        information_schema_column_rows_for_tables(
            &session,
            &[
                "people".to_string(),
                "missing".to_string(),
                "people".to_string(),
                "teams".to_string()
            ]
        ),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                Some("integer".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("name".to_string()),
                Some("2".to_string()),
                Some("text".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                Some("integer".to_string()),
            ],
        ]
    );
    assert_eq!(
        information_schema_rich_columns_query(),
        "select table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
    );
    assert_eq!(
        information_schema_column_details_query_table(
            "select column_name, data_type, is_nullable, column_default from information_schema.columns where table_schema = 'public' and table_name = 'people' order by ordinal_position"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        information_schema_column_detail_rows(&session, "people"),
        vec![
            vec![
                Some("id".to_string()),
                Some("integer".to_string()),
                Some("YES".to_string()),
                None,
            ],
            vec![
                Some("name".to_string()),
                Some("text".to_string()),
                Some("YES".to_string()),
                None,
            ],
        ]
    );
    assert_eq!(
        information_schema_rich_column_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                None,
                Some("YES".to_string()),
                Some("integer".to_string()),
                Some("pg_catalog".to_string()),
                Some("int4".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("name".to_string()),
                Some("2".to_string()),
                None,
                Some("YES".to_string()),
                Some("text".to_string()),
                Some("pg_catalog".to_string()),
                Some("text".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                None,
                Some("YES".to_string()),
                Some("integer".to_string()),
                Some("pg_catalog".to_string()),
                Some("int4".to_string()),
            ],
        ]
    );
    assert_eq!(
        information_schema_extended_columns_query(),
        "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
    );
    assert_eq!(
        information_schema_extended_columns_query_table(
            "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name = 'people' order by ordinal_position"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        information_schema_extended_columns_catalog_query_table(
            "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = current_database() and table_schema = 'public' and table_name = 'people' order by ordinal_position"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        information_schema_extended_columns_catalog_query_table(
            "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = 'postgres' and table_schema = 'public' and table_name = 'people' order by ordinal_position"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        information_schema_extended_columns_in_query_tables(
            "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name in ('teams', 'missing', 'people') order by table_name, ordinal_position"
        ),
        Some(vec![
            "teams".to_string(),
            "missing".to_string(),
            "people".to_string(),
        ])
    );
    assert_eq!(
        information_schema_numeric_metadata(SqlType::Int4),
        (Some(32), Some(2), Some(0))
    );
    assert_eq!(
        information_schema_numeric_metadata(SqlType::Text),
        (None, None, None)
    );
    assert_eq!(
        information_schema_extended_column_rows(&session),
        vec![
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                None,
                Some("YES".to_string()),
                Some("integer".to_string()),
                None,
                Some("32".to_string()),
                Some("2".to_string()),
                Some("0".to_string()),
                Some("pg_catalog".to_string()),
                Some("int4".to_string()),
            ],
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("name".to_string()),
                Some("2".to_string()),
                None,
                Some("YES".to_string()),
                Some("text".to_string()),
                None,
                None,
                None,
                None,
                Some("pg_catalog".to_string()),
                Some("text".to_string()),
            ],
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                None,
                Some("YES".to_string()),
                Some("integer".to_string()),
                None,
                Some("32".to_string()),
                Some("2".to_string()),
                Some("0".to_string()),
                Some("pg_catalog".to_string()),
                Some("int4".to_string()),
            ],
        ]
    );
    assert_eq!(
        information_schema_extended_column_rows_for_table(&session, "people"),
        vec![
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                None,
                Some("YES".to_string()),
                Some("integer".to_string()),
                None,
                Some("32".to_string()),
                Some("2".to_string()),
                Some("0".to_string()),
                Some("pg_catalog".to_string()),
                Some("int4".to_string()),
            ],
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("name".to_string()),
                Some("2".to_string()),
                None,
                Some("YES".to_string()),
                Some("text".to_string()),
                None,
                None,
                None,
                None,
                Some("pg_catalog".to_string()),
                Some("text".to_string()),
            ],
        ]
    );
    assert!(information_schema_extended_column_rows_for_table(&session, "missing").is_empty());
    assert_eq!(
        information_schema_extended_column_rows_for_tables(
            &session,
            &[
                "teams".to_string(),
                "missing".to_string(),
                "people".to_string(),
                "teams".to_string(),
            ],
        ),
        vec![
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                None,
                Some("YES".to_string()),
                Some("integer".to_string()),
                None,
                Some("32".to_string()),
                Some("2".to_string()),
                Some("0".to_string()),
                Some("pg_catalog".to_string()),
                Some("int4".to_string()),
            ],
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("name".to_string()),
                Some("2".to_string()),
                None,
                Some("YES".to_string()),
                Some("text".to_string()),
                None,
                None,
                None,
                None,
                Some("pg_catalog".to_string()),
                Some("text".to_string()),
            ],
            vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("teams".to_string()),
                Some("id".to_string()),
                Some("1".to_string()),
                None,
                Some("YES".to_string()),
                Some("integer".to_string()),
                None,
                Some("32".to_string()),
                Some("2".to_string()),
                Some("0".to_string()),
                Some("pg_catalog".to_string()),
                Some("int4".to_string()),
            ],
        ]
    );
    assert_eq!(
        information_schema_schemata_query(),
        "select schema_name, schema_owner from information_schema.schemata where schema_name = 'public' order by schema_name"
    );
    assert_eq!(
        information_schema_schemata_rows(&Session::default()),
        vec![vec![
            Some("public".to_string()),
            Some("postgres".to_string())
        ]]
    );
    assert_eq!(
        pg_catalog_namespace_query(),
        "select oid, nspname from pg_catalog.pg_namespace where nspname = 'public' order by oid"
    );
    assert_eq!(
        pg_catalog_namespace_rows(&Session::default()),
        vec![vec![
            Some(PUBLIC_NAMESPACE_OID.to_string()),
            Some("public".to_string())
        ]]
    );
    assert_eq!(
        information_schema_table_constraints_query(),
        "select table_schema, table_name, constraint_name, constraint_type from information_schema.table_constraints where table_schema = 'public' order by table_name, constraint_name"
    );
    assert!(information_schema_table_constraint_rows(&session).is_empty());
    assert_eq!(
        information_schema_key_column_usage_query(),
        "select table_schema, table_name, column_name, constraint_name, ordinal_position from information_schema.key_column_usage where table_schema = 'public' order by table_name, ordinal_position"
    );
    assert!(information_schema_key_column_usage_rows(&session).is_empty());
    assert_eq!(
        information_schema_views_query(),
        "select table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into, is_trigger_updatable, is_trigger_deletable, is_trigger_insertable from information_schema.views where table_schema = 'public' order by table_name"
    );
    assert!(information_schema_view_rows(&session).is_empty());
    assert_eq!(
        pg_catalog_views_query(),
        "select schemaname, viewname, viewowner, definition from pg_catalog.pg_views where schemaname = 'public' order by viewname"
    );
    assert!(pg_catalog_view_rows(&session).is_empty());
    assert_eq!(
        pg_catalog_constraints_query(),
        "select n.nspname, c.relname, con.conname, con.contype from pg_catalog.pg_constraint con join pg_catalog.pg_class c on c.oid = con.conrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' order by c.relname, con.conname"
    );
    assert!(pg_catalog_constraint_rows(&session).is_empty());
    assert_eq!(
        pg_catalog_attrdefs_query(),
        "select n.nspname, c.relname, a.attname, pg_catalog.pg_get_expr(d.adbin, d.adrelid) as default_expr from pg_catalog.pg_attrdef d join pg_catalog.pg_class c on c.oid = d.adrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace join pg_catalog.pg_attribute a on a.attrelid = d.adrelid and a.attnum = d.adnum where n.nspname = 'public' order by c.relname, a.attnum"
    );
    assert!(pg_catalog_attrdef_rows(&session).is_empty());
    assert_eq!(
        catalog_type_rows_by_oid(),
        vec![
            vec![
                Some("16".to_string()),
                Some("bool".to_string()),
                Some("1".to_string()),
            ],
            vec![
                Some("20".to_string()),
                Some("int8".to_string()),
                Some("8".to_string()),
            ],
            vec![
                Some("21".to_string()),
                Some("int2".to_string()),
                Some("2".to_string()),
            ],
            vec![
                Some("23".to_string()),
                Some("int4".to_string()),
                Some("4".to_string()),
            ],
            vec![
                Some("25".to_string()),
                Some("text".to_string()),
                Some("-1".to_string()),
            ],
            vec![
                Some("1082".to_string()),
                Some("date".to_string()),
                Some("4".to_string()),
            ],
            vec![
                Some("1114".to_string()),
                Some("timestamp".to_string()),
                Some("8".to_string()),
            ],
            vec![
                Some("1700".to_string()),
                Some("numeric".to_string()),
                Some("-1".to_string()),
            ],
            vec![
                Some("2950".to_string()),
                Some("uuid".to_string()),
                Some("16".to_string()),
            ],
        ]
    );
    assert_eq!(
        catalog_type_rows_by_name(),
        vec![
            vec![
                Some("bool".to_string()),
                Some("16".to_string()),
                Some("1".to_string()),
            ],
            vec![
                Some("date".to_string()),
                Some("1082".to_string()),
                Some("4".to_string()),
            ],
            vec![
                Some("int2".to_string()),
                Some("21".to_string()),
                Some("2".to_string()),
            ],
            vec![
                Some("int4".to_string()),
                Some("23".to_string()),
                Some("4".to_string()),
            ],
            vec![
                Some("int8".to_string()),
                Some("20".to_string()),
                Some("8".to_string()),
            ],
            vec![
                Some("numeric".to_string()),
                Some("1700".to_string()),
                Some("-1".to_string()),
            ],
            vec![
                Some("text".to_string()),
                Some("25".to_string()),
                Some("-1".to_string()),
            ],
            vec![
                Some("timestamp".to_string()),
                Some("1114".to_string()),
                Some("8".to_string()),
            ],
            vec![
                Some("uuid".to_string()),
                Some("2950".to_string()),
                Some("16".to_string()),
            ],
        ]
    );
    assert_eq!(
        catalog_attribute_query_table(
            "select attname, atttypid from pg_catalog.pg_attribute where attrelid = 'people'::regclass and attnum > 0 order by attnum"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        catalog_attribute_rows(&session, "people").unwrap(),
        vec![
            vec![Some("id".to_string()), Some("23".to_string())],
            vec![Some("name".to_string()), Some("25".to_string())],
        ]
    );
    assert_eq!(
        catalog_attribute_detail_query_table(
            "select attnum, attname, atttypid, attlen from pg_catalog.pg_attribute where attrelid = 'people'::regclass and attnum > 0 order by attnum"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        catalog_attribute_detail_rows(&session, "people").unwrap(),
        vec![
            vec![
                Some("1".to_string()),
                Some("id".to_string()),
                Some("23".to_string()),
                Some("4".to_string()),
            ],
            vec![
                Some("2".to_string()),
                Some("name".to_string()),
                Some("25".to_string()),
                Some("-1".to_string()),
            ],
        ]
    );
    assert_eq!(
        pg_catalog_class_attribute_type_query_table(
            "select a.attnum, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type, a.attnotnull from pg_catalog.pg_attribute a join pg_catalog.pg_class c on c.oid = a.attrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname = 'people' and a.attnum > 0 and not a.attisdropped order by a.attnum"
        ),
        Some("people".to_string())
    );
    assert_eq!(
        pg_catalog_class_attribute_type_rows(&session, "people").unwrap(),
        vec![
            vec![
                Some("1".to_string()),
                Some("id".to_string()),
                Some("integer".to_string()),
                Some("f".to_string()),
            ],
            vec![
                Some("2".to_string()),
                Some("name".to_string()),
                Some("text".to_string()),
                Some("f".to_string()),
            ],
        ]
    );
    assert!(catalog_attribute_rows(&session, "missing").is_none());
}

#[test]
fn catalog_introspection_helpers_expose_relation_oids_and_attribute_details() {
    let mut session = Session::default();
    session.tables.insert(
        "people".to_string(),
        Table {
            oid: FIRST_USER_RELATION_OID,
            name: "people".to_string(),
            columns: vec![CatalogColumn {
                attnum: 1,
                def: gpu_db_protocol::ColumnDef {
                    name: "id".to_string(),
                    ty: gpu_db_protocol::SqlType::Int4,
                    domain: None,
                    default: None,
                },
            }],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );

    assert_eq!(
        catalog_table_oid_rows(&session),
        vec![vec![
            Some(FIRST_USER_RELATION_OID.to_string()),
            Some("people".to_string()),
        ]]
    );
    assert_eq!(
        catalog_attribute_detail_rows(&session, "people").unwrap(),
        vec![vec![
            Some("1".to_string()),
            Some("id".to_string()),
            Some("23".to_string()),
            Some("4".to_string()),
        ]]
    );
}

#[test]
fn row_filtering_honors_disjunctive_select_groups() {
    let table = Table {
        oid: FIRST_USER_RELATION_OID,
        name: "people".to_string(),
        columns: vec![
            CatalogColumn {
                attnum: 1,
                def: gpu_db_protocol::ColumnDef {
                    name: "id".to_string(),
                    ty: gpu_db_protocol::SqlType::Int4,
                    domain: None,
                    default: None,
                },
            },
            CatalogColumn {
                attnum: 2,
                def: gpu_db_protocol::ColumnDef {
                    name: "name".to_string(),
                    ty: gpu_db_protocol::SqlType::Text,
                    domain: None,
                    default: None,
                },
            },
        ],
        rows: Vec::new(),
        check_constraints: Vec::new(),
        foreign_keys: Vec::new(),
    };
    let Command::Select(select) =
        parse_command("SELECT id, name FROM people WHERE (id = 1) OR (name = 'Grace')").unwrap()
    else {
        panic!("expected SELECT plan");
    };

    assert!(row_matches_select_filters(
        &table,
        &[SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
        &select,
    )
    .unwrap());
    assert!(row_matches_select_filters(
        &table,
        &[SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
        &select,
    )
    .unwrap());
    assert!(!row_matches_select_filters(
        &table,
        &[SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
        &select,
    )
    .unwrap());
}

#[test]
fn describe_query_columns_handles_parameterized_select_shapes() {
    let mut session = Session::default();
    session.tables.insert(
        "people".to_string(),
        Table {
            oid: FIRST_USER_RELATION_OID,
            name: "people".to_string(),
            columns: vec![
                CatalogColumn {
                    attnum: 1,
                    def: gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: gpu_db_protocol::SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: gpu_db_protocol::SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );

    assert_eq!(
        describe_query_columns(
            &session,
            "SELECT name, id FROM people WHERE id = $1 ORDER BY name DESC LIMIT 1",
        ),
        Some(vec![text_column("name"), int4_column("id")])
    );
    assert_eq!(
        describe_query_columns(
            &session,
            "SELECT name, id FROM people WHERE id = $1 ORDER BY name DESC LIMIT -1",
        ),
        Some(vec![text_column("name"), int4_column("id")])
    );
}

#[test]
fn psql_gdesc_type_rows_formats_supported_row_description_types() {
    assert_eq!(
        psql_describe_query_type_rows(
            "select name as \"column\", pg_catalog.format_type(tp, tpm) as \"type\" from (values ('name', '25'::pg_catalog.oid, -1),('id', '23'::pg_catalog.oid, -1)) s(name, tp, tpm)"
        ),
        Some(vec![
            vec![Some("name".to_string()), Some("text".to_string())],
            vec![Some("id".to_string()), Some("integer".to_string())],
        ])
    );
    assert_eq!(
        psql_describe_query_type_rows(
            "select name as \"column\", pg_catalog.format_type(tp, tpm) as \"type\" from (values ('unsupported', '999999'::pg_catalog.oid, -1)) s(name, tp, tpm)"
        ),
        None
    );
}
