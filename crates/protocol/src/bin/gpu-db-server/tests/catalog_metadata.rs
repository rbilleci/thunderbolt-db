use super::*;

#[test]
fn catalog_helpers_parse_catalog_qualified_information_schema_table_filters() {
    let current_database_query = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_catalog = current_database() and table_schema = 'public' and table_name = 'people' order by table_name";
    let literal_catalog_query = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_catalog = 'postgres' and table_schema = 'public' and table_name = 'people' order by table_name";

    assert_eq!(
        information_schema_rich_tables_catalog_query_table(current_database_query),
        Some("people".to_string())
    );
    assert_eq!(
        information_schema_rich_tables_catalog_query_table(literal_catalog_query),
        Some("people".to_string())
    );
}

#[test]
fn catalog_pg_namespace_pg_dump_public_schema_discovery_queries() {
    assert!(is_pg_dump_public_namespace_oid_lookup_query(
        "select oid from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default"
    ));

    let mut session = Session::default();
    let (mut writer, mut reader) = tcp_pair();
    execute_statement(&mut writer, &mut session, "CREATE SCHEMA public", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
}

#[test]
fn catalog_pg_dump_custom_archive_database_metadata_query() {
    assert_eq!(
        canonical_sql(
            "SELECT tableoid, oid, datname, datdba, pg_encoding_to_char(encoding) AS encoding, datcollate, datctype, datfrozenxid, datacl, acldefault('d', datdba) AS acldefault, datistemplate, datconnlimit, datminmxid, datlocprovider, daticulocale, datcollversion, daticurules, (SELECT spcname FROM pg_tablespace t WHERE t.oid = dattablespace) AS tablespace, shobj_description(oid, 'pg_database') AS description FROM pg_database WHERE datname = current_database()"
        ),
        pg_dump_database_metadata_query()
    );
    assert_eq!(
        pg_dump_database_metadata_rows(&Session::default())[0][2],
        Some("postgres".to_string())
    );
    let mut session = Session::default();
    session.comments.insert(
        CatalogCommentTarget::Database {
            database: "postgres".to_string(),
        },
        "primary database".to_string(),
    );
    assert_eq!(
        pg_dump_database_metadata_rows(&session)[0][18],
        Some("primary database".to_string())
    );
}

#[test]
fn catalog_pg_dump_column_acl_discovery_returns_empty() {
    assert_eq!(
        pg_dump_empty_catalog_query_columns(&canonical_sql(
            "SELECT DISTINCT attrelid FROM pg_attribute WHERE attacl IS NOT NULL"
        )),
        Some(vec![int4_column("attrelid")])
    );
    assert_eq!(
        pg_dump_empty_catalog_query_columns(&canonical_sql(
            "SELECT objoid, classoid, objsubid, privtype, initprivs FROM pg_init_privs"
        )),
        Some(vec![
            int4_column("objoid"),
            int4_column("classoid"),
            int4_column("objsubid"),
            text_column("privtype"),
            text_column("initprivs"),
        ])
    );
}

#[test]
fn catalog_queries_expose_literal_column_defaults() {
    let mut session = Session::default();
    session.tables.insert(
        "defaults".to_string(),
        Table {
            oid: FIRST_USER_RELATION_OID,
            name: "defaults".to_string(),
            columns: vec![
                CatalogColumn {
                    attnum: 1,
                    def: gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: Some(ColumnDefault::Literal(SqlValue::Int4(7))),
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: Some(ColumnDefault::Literal(SqlValue::Text("Ada's".to_string()))),
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );

    assert_eq!(
        information_schema_column_detail_rows(&session, "defaults"),
        vec![
            vec![
                Some("id".to_string()),
                Some("integer".to_string()),
                Some("YES".to_string()),
                Some("7".to_string()),
            ],
            vec![
                Some("name".to_string()),
                Some("text".to_string()),
                Some("YES".to_string()),
                Some("'Ada''s'::text".to_string()),
            ],
        ]
    );
    assert_eq!(
        catalog_describe_attribute_rows(&session, FIRST_USER_RELATION_OID),
        vec![
            vec![
                Some("id".to_string()),
                Some("integer".to_string()),
                Some("7".to_string()),
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
            ],
            vec![
                Some("name".to_string()),
                Some("text".to_string()),
                Some("'Ada''s'::text".to_string()),
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
            ],
        ]
    );
    assert_eq!(
        pg_catalog_attrdef_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("defaults".to_string()),
                Some("id".to_string()),
                Some("7".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("defaults".to_string()),
                Some("name".to_string()),
                Some("'Ada''s'::text".to_string()),
            ],
        ]
    );
}

#[test]
fn catalog_queries_expose_table_and_column_comments() {
    let mut session = Session::default();
    session.comments.insert(
        CatalogCommentTarget::Database {
            database: "postgres".to_string(),
        },
        "primary database".to_string(),
    );
    session.comments.insert(
        CatalogCommentTarget::Role {
            role: "postgres".to_string(),
        },
        "bootstrap role".to_string(),
    );
    session.comments.insert(
        CatalogCommentTarget::Schema {
            schema: "public".to_string(),
        },
        "application schema".to_string(),
    );
    session.comments.insert(
        CatalogCommentTarget::Tablespace {
            tablespace: "pg_default".to_string(),
        },
        "default storage".to_string(),
    );
    session.tables.insert(
        "commented".to_string(),
        Table {
            oid: FIRST_USER_RELATION_OID,
            name: "commented".to_string(),
            columns: vec![
                CatalogColumn {
                    attnum: 1,
                    def: gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
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
    session.comments.insert(
        CatalogCommentTarget::Table {
            table: "commented".to_string(),
        },
        "lookup table".to_string(),
    );
    session.comments.insert(
        CatalogCommentTarget::Column {
            table: "commented".to_string(),
            attnum: 2,
        },
        "display name".to_string(),
    );
    let view_query =
        match parse_command("SELECT id, name FROM commented WHERE id > 0 ORDER BY id").unwrap() {
            Command::Select(select) => select,
            _ => panic!("expected SELECT"),
        };
    session.views.insert(
        "commented_view".to_string(),
        View {
            oid: FIRST_USER_RELATION_OID + 1,
            name: "commented_view".to_string(),
            query: view_query,
            definition: "SELECT id, name FROM commented WHERE id > 0 ORDER BY id".to_string(),
        },
    );
    session.comments.insert(
        CatalogCommentTarget::View {
            view: "commented_view".to_string(),
        },
        "lookup view".to_string(),
    );

    assert_eq!(
        catalog_psql_describe_role_rows(&session, true)[0][8],
        Some("bootstrap role".to_string())
    );
    assert_eq!(
        catalog_psql_list_tablespace_rows(&session, true)[0][6],
        Some("default storage".to_string())
    );
    assert_eq!(
        catalog_psql_describe_schema_verbose_rows(&session),
        vec![vec![
            Some("public".to_string()),
            Some("postgres".to_string()),
            None,
            Some("application schema".to_string()),
        ]]
    );
    assert_eq!(
        pg_catalog_schema_description_rows(&session),
        vec![vec![
            Some("public".to_string()),
            Some("application schema".to_string()),
        ]]
    );
    assert_eq!(
        pg_catalog_description_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("commented".to_string()),
                None,
                Some("lookup table".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("commented".to_string()),
                Some("name".to_string()),
                Some("display name".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("commented_view".to_string()),
                None,
                Some("lookup view".to_string()),
            ],
        ]
    );
    assert_eq!(
        pg_dump_description_rows(&session),
        vec![
            vec![
                Some("lookup table".to_string()),
                Some("1259".to_string()),
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("0".to_string()),
            ],
            vec![
                Some("display name".to_string()),
                Some("1259".to_string()),
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("2".to_string()),
            ],
            vec![
                Some("lookup view".to_string()),
                Some("1259".to_string()),
                Some((FIRST_USER_RELATION_OID + 1).to_string()),
                Some("0".to_string()),
            ],
            vec![
                Some("application schema".to_string()),
                Some("2615".to_string()),
                Some(PUBLIC_NAMESPACE_OID.to_string()),
                Some("0".to_string()),
            ],
        ]
    );
    assert_eq!(
        psql_object_description_rows(&session),
        vec![
            vec![
                Some("public".to_string()),
                Some("public".to_string()),
                Some("schema".to_string()),
                Some("application schema".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("commented".to_string()),
                Some("table".to_string()),
                Some("lookup table".to_string()),
            ],
            vec![
                Some("public".to_string()),
                Some("commented_view".to_string()),
                Some("view".to_string()),
                Some("lookup view".to_string()),
            ],
        ]
    );
}
