use super::*;

#[test]
fn catalog_index_rows_reflect_created_indexes() {
    let mut session = Session::default();
    session
        .tables
        .insert("people".to_string(), test_table("people", Vec::new()));
    session.indexes.push(CatalogIndex {
        name: "people_name_idx".to_string(),
        table: "people".to_string(),
        column: "id".to_string(),
        unique: false,
        primary_key: false,
        unique_constraint: false,
    });
    session.comments.insert(
        CatalogCommentTarget::Index {
            index: "people_name_idx".to_string(),
        },
        "lookup index".to_string(),
    );

    assert_eq!(
        pg_catalog_index_rows(&session),
        vec![vec![
            Some("public".to_string()),
            Some("people".to_string()),
            Some("people_name_idx".to_string()),
            Some("CREATE INDEX people_name_idx ON public.people USING btree (id)".to_string()),
        ]]
    );
    assert_eq!(
        psql_describe_index_rows(&session),
        vec![vec![
            Some("public".to_string()),
            Some("people_name_idx".to_string()),
            Some("index".to_string()),
            Some("postgres".to_string()),
            Some("people".to_string()),
        ]]
    );
    assert_eq!(
        psql_describe_index_verbose_rows(&session),
        vec![vec![
            Some("public".to_string()),
            Some("people_name_idx".to_string()),
            Some("index".to_string()),
            Some("postgres".to_string()),
            Some("people".to_string()),
            Some("permanent".to_string()),
            Some("btree".to_string()),
            None,
            Some("lookup index".to_string()),
        ]]
    );
    assert_eq!(
        pg_dump_description_rows(&session),
        vec![vec![
            Some("lookup index".to_string()),
            Some("1259".to_string()),
            Some(FIRST_USER_INDEX_OID.to_string()),
            Some("0".to_string()),
        ]]
    );
    assert_eq!(
        pg_dump_index_metadata_rows(&session),
        vec![vec![
            Some("1259".to_string()),
            Some(FIRST_USER_INDEX_OID.to_string()),
            Some(FIRST_USER_RELATION_OID.to_string()),
            Some("people_name_idx".to_string()),
            Some("CREATE INDEX people_name_idx ON public.people USING btree (id)".to_string()),
            Some("1".to_string()),
            Some("f".to_string()),
            None,
            None,
            Some("f".to_string()),
            Some("f".to_string()),
            None,
            None,
            None,
            Some(String::new()),
            None,
            Some("f".to_string()),
            Some("0".to_string()),
            Some("1".to_string()),
            Some("1".to_string()),
            None,
            None,
            Some("f".to_string()),
        ]]
    );
}

#[test]
fn catalog_extension_rows_reflect_supported_comment() {
    let mut session = Session::default();
    session.comments.insert(
        CatalogCommentTarget::Extension {
            extension: "plpgsql".to_string(),
        },
        "bootstrap extension".to_string(),
    );

    assert_eq!(
        catalog_psql_extension_rows(&session),
        vec![vec![
            Some("plpgsql".to_string()),
            Some("1.0".to_string()),
            Some("pg_catalog".to_string()),
            Some("bootstrap extension".to_string()),
        ]]
    );
    assert!(pg_dump_description_rows(&session).contains(&vec![
        Some("bootstrap extension".to_string()),
        Some(PG_EXTENSION_CLASS_OID.to_string()),
        Some(PLPGSQL_EXTENSION_OID.to_string()),
        Some("0".to_string()),
    ]));
}

#[test]
fn shared_catalog_persistence_carries_index_metadata() {
    let table_name = "shared_index_people";
    let index_name = "shared_index_people_id_idx";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tables.remove(table_name);
        catalog.indexes.retain(|index| index.name != index_name);
    }

    let mut session = Session::new(true);
    session
        .tables
        .insert(table_name.to_string(), test_table(table_name, Vec::new()));
    session.indexes.push(CatalogIndex {
        name: index_name.to_string(),
        table: table_name.to_string(),
        column: "id".to_string(),
        unique: false,
        primary_key: false,
        unique_constraint: false,
    });
    session.mark_table_dirty(table_name);
    session.dirty_indexes = true;
    session.persist_catalog_snapshot();

    let reloaded = Session::new(true);
    assert!(pg_catalog_index_rows(&reloaded).contains(&vec![
        Some("public".to_string()),
        Some(table_name.to_string()),
        Some(index_name.to_string()),
        Some(format!(
            "CREATE INDEX {index_name} ON public.{table_name} USING btree (id)"
        )),
    ]));

    let mut catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    catalog.tables.remove(table_name);
    catalog.indexes.retain(|index| index.name != index_name);
}

#[test]
fn shared_catalog_persistence_carries_default_table_acl_metadata() {
    let table_name = "shared_default_acl_people";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tables.remove(table_name);
        catalog.table_acls.remove(table_name);
        catalog.default_table_acl.clear();
    }

    let mut session = Session::new(true);
    grant_default_table_acl(&mut session, "public", &[TablePrivilege::Select]).unwrap();
    session.persist_catalog_snapshot();

    let mut reloaded = Session::new(true);
    reloaded
        .tables
        .insert(table_name.to_string(), test_table(table_name, Vec::new()));
    if !reloaded.default_table_acl.is_empty() {
        reloaded
            .table_acls
            .insert(table_name.to_string(), reloaded.default_table_acl.clone());
        reloaded.mark_table_acl_dirty(table_name);
    }
    reloaded.mark_table_dirty(table_name);
    reloaded.persist_catalog_snapshot();

    let final_session = Session::new(true);
    assert_eq!(
        relation_acl_display(&final_session, table_name),
        Some("=r/postgres".to_string())
    );
    assert_eq!(
        catalog_psql_default_access_privilege_rows(&final_session),
        vec![vec![
            Some("postgres".to_string()),
            Some("public".to_string()),
            Some("table".to_string()),
            Some("=r/postgres".to_string()),
        ]]
    );

    let mut catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    catalog.tables.remove(table_name);
    catalog.table_acls.remove(table_name);
    catalog.default_table_acl.clear();
}

#[test]
fn relation_acl_enforcement_uses_current_role_and_public_grants() {
    let mut session = Session::default();
    session.roles.insert(
        "reader".to_string(),
        RoleInfo {
            oid: FIRST_USER_RELATION_OID + 1,
            name: "reader".to_string(),
            login: false,
        },
    );
    session.tables.insert(
        "acl_people".to_string(),
        test_table(
            "acl_people",
            vec![
                vec![SqlValue::Int4(1)],
                vec![SqlValue::Int4(2)],
                vec![SqlValue::Int4(3)],
            ],
        ),
    );
    let Command::Select(select) = parse_command("select id from acl_people order by id").unwrap()
    else {
        panic!("expected supported SELECT");
    };

    session.current_role = Some("reader".to_string());
    let err = execute_select_result(&session, &select).unwrap_err();
    assert_eq!(err.code, "42501");
    assert_eq!(err.message, "permission denied for relation");

    grant_relation_acl(
        &mut session,
        "acl_people",
        AclRelationKind::Table,
        "public",
        &[TablePrivilege::Select],
    )
    .unwrap();
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![Some("1".to_string())],
            vec![Some("2".to_string())],
            vec![Some("3".to_string())],
        ]
    );

    session.current_role = None;
    revoke_relation_acl(
        &mut session,
        "acl_people",
        AclRelationKind::Table,
        "public",
        &[TablePrivilege::Select],
    )
    .unwrap();
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(result.rows.len(), 3);
}

#[test]
fn schema_usage_enforcement_gates_supported_object_access() {
    let mut session = Session::default();
    session.roles.insert(
        "reader".to_string(),
        RoleInfo {
            oid: FIRST_USER_RELATION_OID + 1,
            name: "reader".to_string(),
            login: false,
        },
    );
    session.tables.insert(
        "acl_people".to_string(),
        test_table("acl_people", vec![vec![SqlValue::Int4(1)]]),
    );
    session.functions.insert(
        "acl_answer".to_string(),
        FunctionInfo {
            oid: FIRST_USER_RELATION_OID + 2,
            name: "acl_answer".to_string(),
            return_type: SqlType::Int4,
            body: "SELECT 42".to_string(),
            acl: BTreeMap::new(),
        },
    );
    grant_relation_acl(
        &mut session,
        "acl_people",
        AclRelationKind::Table,
        "reader",
        &[TablePrivilege::Select],
    )
    .unwrap();
    grant_schema_acl(&mut session, "public", "reader", &[SchemaPrivilege::Create]).unwrap();
    session.current_role = Some("reader".to_string());

    let Command::Select(select) = parse_command("select id from acl_people order by id").unwrap()
    else {
        panic!("expected supported SELECT");
    };
    let err = execute_select_result(&session, &select).unwrap_err();
    assert_eq!(err.code, "42501");
    assert_eq!(err.message, "permission denied for schema");

    let Command::SelectFunction(call) = parse_command("select acl_answer()").unwrap() else {
        panic!("expected supported function call");
    };
    let err = execute_function_result(&session, &call).unwrap_err();
    assert_eq!(err.code, "42501");
    assert_eq!(err.message, "permission denied for schema");

    grant_schema_acl(&mut session, "public", "reader", &[SchemaPrivilege::Usage]).unwrap();
    assert_eq!(
        execute_select_result(&session, &select).unwrap().rows.len(),
        1
    );
    let err = execute_function_result(&session, &call).unwrap_err();
    assert_eq!(err.code, "42501");
    assert_eq!(err.message, "permission denied for function");
    grant_function_acl(
        &mut session,
        "acl_answer",
        "reader",
        &[FunctionPrivilege::Execute],
    )
    .unwrap();
    assert_eq!(
        execute_function_result(&session, &call).unwrap().rows,
        vec![vec![Some("42".to_string())]]
    );
}

#[test]
fn shared_catalog_persistence_carries_database_metadata() {
    let database_name = "shared_appdb";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.databases.remove(database_name);
        catalog.database_acls.remove(database_name);
        catalog.comments.remove(&CatalogCommentTarget::Database {
            database: database_name.to_string(),
        });
    }

    let mut session = Session::new(true);
    session.databases.insert(
        database_name.to_string(),
        DatabaseInfo {
            oid: FIRST_USER_RELATION_OID,
            name: database_name.to_string(),
        },
    );
    session.comments.insert(
        CatalogCommentTarget::Database {
            database: database_name.to_string(),
        },
        "shared database".to_string(),
    );
    session.database_acls.insert(
        database_name.to_string(),
        BTreeMap::from([(
            "public".to_string(),
            BTreeSet::from([DatabasePrivilege::Connect]),
        )]),
    );
    session.mark_database_dirty(database_name);
    session.mark_database_acl_dirty(database_name);
    session.mark_comment_dirty(CatalogCommentTarget::Database {
        database: database_name.to_string(),
    });
    session.persist_catalog_snapshot();

    let mut reloaded = Session::new(true);
    assert_eq!(
        catalog_database_oid_rows(&reloaded),
        vec![
            vec![
                Some(POSTGRES_DATABASE_OID.to_string()),
                Some("postgres".to_string()),
            ],
            vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some(database_name.to_string()),
            ],
        ]
    );
    assert_eq!(
        catalog_psql_list_database_verbose_rows(&reloaded)[1][11],
        Some("shared database".to_string())
    );
    assert_eq!(
        catalog_psql_list_database_verbose_rows(&reloaded)[1][8],
        Some("=c/postgres".to_string())
    );

    reloaded.databases.remove(database_name);
    reloaded.database_acls.remove(database_name);
    reloaded.comments.remove(&CatalogCommentTarget::Database {
        database: database_name.to_string(),
    });
    reloaded.mark_database_dirty(database_name);
    reloaded.mark_database_acl_dirty(database_name);
    reloaded.mark_comment_dirty(CatalogCommentTarget::Database {
        database: database_name.to_string(),
    });
    reloaded.persist_catalog_snapshot();

    let final_session = Session::new(true);
    assert!(!final_session.databases.contains_key(database_name));
    assert!(!final_session.database_acls.contains_key(database_name));
    assert!(!final_session
        .comments
        .contains_key(&CatalogCommentTarget::Database {
            database: database_name.to_string(),
        }));
}

#[test]
fn shared_catalog_persistence_carries_tablespace_metadata() {
    let tablespace_name = "shared_appspace";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tablespaces.remove(tablespace_name);
        catalog.tablespace_acls.remove(tablespace_name);
        catalog.comments.remove(&CatalogCommentTarget::Tablespace {
            tablespace: tablespace_name.to_string(),
        });
    }

    let mut session = Session::new(true);
    session.tablespaces.insert(
        tablespace_name.to_string(),
        TablespaceInfo {
            oid: FIRST_USER_RELATION_OID,
            name: tablespace_name.to_string(),
            location: "/tmp/shared_appspace".to_string(),
        },
    );
    session.comments.insert(
        CatalogCommentTarget::Tablespace {
            tablespace: tablespace_name.to_string(),
        },
        "shared storage".to_string(),
    );
    session.tablespace_acls.insert(
        tablespace_name.to_string(),
        BTreeMap::from([(
            "public".to_string(),
            BTreeSet::from([TablespacePrivilege::Create]),
        )]),
    );
    session.mark_tablespace_dirty(tablespace_name);
    session.mark_tablespace_acl_dirty(tablespace_name);
    session.mark_comment_dirty(CatalogCommentTarget::Tablespace {
        tablespace: tablespace_name.to_string(),
    });
    session.persist_catalog_snapshot();

    let mut reloaded = Session::new(true);
    let oid_rows = catalog_tablespace_oid_rows(&reloaded);
    assert!(oid_rows.contains(&vec![
        Some(FIRST_USER_RELATION_OID.to_string()),
        Some(tablespace_name.to_string()),
        Some("/tmp/shared_appspace".to_string()),
    ]));
    let verbose_rows = catalog_psql_list_tablespace_rows(&reloaded, true);
    let appspace_row = verbose_rows
        .iter()
        .find(|row| row[0] == Some(tablespace_name.to_string()))
        .expect("shared tablespace row");
    assert_eq!(appspace_row[6], Some("shared storage".to_string()));
    assert_eq!(appspace_row[3], Some("=C/postgres".to_string()));
    assert_eq!(
        pg_dumpall_tablespace_metadata_rows(&reloaded),
        vec![vec![
            Some(FIRST_USER_RELATION_OID.to_string()),
            Some(tablespace_name.to_string()),
            Some("postgres".to_string()),
            Some("/tmp/shared_appspace".to_string()),
            Some("{postgres=C/postgres,=C/postgres}".to_string()),
            Some("{postgres=C/postgres}".to_string()),
            None,
            Some("shared storage".to_string()),
        ]]
    );

    reloaded.tablespaces.remove(tablespace_name);
    reloaded.tablespace_acls.remove(tablespace_name);
    reloaded.comments.remove(&CatalogCommentTarget::Tablespace {
        tablespace: tablespace_name.to_string(),
    });
    reloaded.mark_tablespace_dirty(tablespace_name);
    reloaded.mark_tablespace_acl_dirty(tablespace_name);
    reloaded.mark_comment_dirty(CatalogCommentTarget::Tablespace {
        tablespace: tablespace_name.to_string(),
    });
    reloaded.persist_catalog_snapshot();

    let final_session = Session::new(true);
    assert!(!final_session.tablespaces.contains_key(tablespace_name));
    assert!(!final_session.tablespace_acls.contains_key(tablespace_name));
    assert!(!final_session
        .comments
        .contains_key(&CatalogCommentTarget::Tablespace {
            tablespace: tablespace_name.to_string(),
        }));
}

#[test]
fn relation_acl_rows_include_supported_views_materialized_views_and_sequences() {
    let mut session = Session::default();
    session.tables.insert(
        "rel_acl_people".to_string(),
        test_table("rel_acl_people", Vec::new()),
    );
    let query = match parse_command("SELECT * FROM rel_acl_people").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected select, got {other:?}"),
    };
    session.views.insert(
        "rel_acl_view".to_string(),
        View {
            oid: FIRST_USER_RELATION_OID + 1,
            name: "rel_acl_view".to_string(),
            query: query.clone(),
            definition: "SELECT * FROM rel_acl_people".to_string(),
        },
    );
    session.materialized_views.insert(
        "rel_acl_mv".to_string(),
        MaterializedView {
            oid: FIRST_USER_RELATION_OID + 2,
            name: "rel_acl_mv".to_string(),
            query,
            definition: "SELECT * FROM rel_acl_people".to_string(),
            columns: Vec::new(),
            rows: Vec::new(),
        },
    );
    session.sequences.insert(
        "rel_acl_seq".to_string(),
        Sequence {
            oid: FIRST_USER_RELATION_OID + 3,
            name: "rel_acl_seq".to_string(),
            last_value: 1,
            is_called: false,
        },
    );
    grant_relation_acl(
        &mut session,
        "rel_acl_view",
        AclRelationKind::View,
        "public",
        &[TablePrivilege::Select],
    )
    .unwrap();
    grant_relation_acl(
        &mut session,
        "rel_acl_mv",
        AclRelationKind::MaterializedView,
        "public",
        &[TablePrivilege::Select],
    )
    .unwrap();
    grant_relation_acl(
        &mut session,
        "rel_acl_seq",
        AclRelationKind::Sequence,
        "postgres",
        &[TablePrivilege::Select, TablePrivilege::Update],
    )
    .unwrap();

    let rows = catalog_psql_describe_table_privilege_rows_filtered(
        &session,
        &PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: None,
        },
    );
    assert_eq!(
        rows,
        vec![
            vec![
                Some("public".to_string()),
                Some("rel_acl_mv".to_string()),
                Some("materialized view".to_string()),
                Some("=r/postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("public".to_string()),
                Some("rel_acl_people".to_string()),
                Some("table".to_string()),
                None,
                None,
                None,
            ],
            vec![
                Some("public".to_string()),
                Some("rel_acl_seq".to_string()),
                Some("sequence".to_string()),
                Some("postgres=rw/postgres".to_string()),
                None,
                None,
            ],
            vec![
                Some("public".to_string()),
                Some("rel_acl_view".to_string()),
                Some("view".to_string()),
                Some("=r/postgres".to_string()),
                None,
                None,
            ],
        ]
    );
}

#[test]
fn shared_catalog_persistence_carries_renamed_index_metadata() {
    let table_name = "shared_rename_index_people";
    let old_index_name = "shared_rename_index_people_id_idx";
    let new_index_name = "shared_rename_index_people_lookup_idx";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tables.remove(table_name);
        catalog
            .indexes
            .retain(|index| index.name != old_index_name && index.name != new_index_name);
        catalog.comments.remove(&CatalogCommentTarget::Index {
            index: old_index_name.to_string(),
        });
        catalog.comments.remove(&CatalogCommentTarget::Index {
            index: new_index_name.to_string(),
        });
    }

    let mut session = Session::new(true);
    session
        .tables
        .insert(table_name.to_string(), test_table(table_name, Vec::new()));
    session.indexes.push(CatalogIndex {
        name: old_index_name.to_string(),
        table: table_name.to_string(),
        column: "id".to_string(),
        unique: false,
        primary_key: false,
        unique_constraint: false,
    });
    session.comments.insert(
        CatalogCommentTarget::Index {
            index: old_index_name.to_string(),
        },
        "lookup index".to_string(),
    );
    session.mark_table_dirty(table_name);
    session.dirty_indexes = true;
    session.persist_catalog_snapshot();

    rename_index_in_session(&mut session, old_index_name, new_index_name).unwrap();

    let reloaded = Session::new(true);
    assert!(pg_catalog_index_rows(&reloaded).contains(&vec![
        Some("public".to_string()),
        Some(table_name.to_string()),
        Some(new_index_name.to_string()),
        Some(format!(
            "CREATE INDEX {new_index_name} ON public.{table_name} USING btree (id)"
        )),
    ]));
    assert!(psql_describe_index_verbose_rows(&reloaded).contains(&vec![
        Some("public".to_string()),
        Some(new_index_name.to_string()),
        Some("index".to_string()),
        Some("postgres".to_string()),
        Some(table_name.to_string()),
        Some("permanent".to_string()),
        Some("btree".to_string()),
        None,
        Some("lookup index".to_string()),
    ]));
    assert!(!pg_catalog_index_rows(&reloaded)
        .iter()
        .any(|row| { row.get(2).and_then(Option::as_deref) == Some(old_index_name) }));

    assert_eq!(
        rename_index_in_session(&mut session, "missing_idx", "another_idx")
            .unwrap_err()
            .code,
        "42704"
    );
    session.indexes.push(CatalogIndex {
        name: "shared_rename_index_people_pkey".to_string(),
        table: table_name.to_string(),
        column: "id".to_string(),
        unique: true,
        primary_key: true,
        unique_constraint: false,
    });
    assert_eq!(
        rename_index_in_session(
            &mut session,
            "shared_rename_index_people_pkey",
            "shared_rename_index_people_id_idx"
        )
        .unwrap_err()
        .code,
        "0A000"
    );

    let mut catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    catalog.tables.remove(table_name);
    catalog.indexes.retain(|index| {
        index.name != old_index_name
            && index.name != new_index_name
            && index.name != "shared_rename_index_people_pkey"
    });
    catalog.comments.remove(&CatalogCommentTarget::Index {
        index: old_index_name.to_string(),
    });
    catalog.comments.remove(&CatalogCommentTarget::Index {
        index: new_index_name.to_string(),
    });
}

#[test]
fn shared_catalog_persistence_carries_renamed_sequence_metadata() {
    let old_sequence_name = "shared_rename_people_seq";
    let new_sequence_name = "shared_renamed_people_seq";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.sequences.remove(old_sequence_name);
        catalog.sequences.remove(new_sequence_name);
        catalog.comments.remove(&CatalogCommentTarget::Sequence {
            sequence: old_sequence_name.to_string(),
        });
        catalog.comments.remove(&CatalogCommentTarget::Sequence {
            sequence: new_sequence_name.to_string(),
        });
    }

    let mut session = Session::new(true);
    session.sequences.insert(
        old_sequence_name.to_string(),
        Sequence {
            oid: FIRST_USER_RELATION_OID,
            name: old_sequence_name.to_string(),
            last_value: 1,
            is_called: false,
        },
    );
    session.comments.insert(
        CatalogCommentTarget::Sequence {
            sequence: old_sequence_name.to_string(),
        },
        "people ids".to_string(),
    );
    session.mark_sequence_dirty(old_sequence_name);
    session.mark_comment_dirty(CatalogCommentTarget::Sequence {
        sequence: old_sequence_name.to_string(),
    });
    session.persist_catalog_snapshot();

    rename_sequence_in_session(&mut session, old_sequence_name, new_sequence_name).unwrap();

    let reloaded = Session::new(true);
    assert!(pg_catalog_class_sequence_rows(&reloaded).contains(&vec![
        Some(FIRST_USER_RELATION_OID.to_string()),
        Some("public".to_string()),
        Some(new_sequence_name.to_string()),
        Some("s".to_string()),
        Some("p".to_string()),
    ]));
    assert!(
        psql_describe_sequence_verbose_rows(&reloaded).contains(&vec![
            Some("public".to_string()),
            Some(new_sequence_name.to_string()),
            Some("sequence".to_string()),
            Some("postgres".to_string()),
            Some("permanent".to_string()),
            Some("0 bytes".to_string()),
            Some("people ids".to_string()),
        ])
    );
    assert!(!pg_catalog_class_sequence_rows(&reloaded)
        .iter()
        .any(|row| row.get(2).and_then(Option::as_deref) == Some(old_sequence_name)));
    assert_eq!(
        rename_sequence_in_session(&mut session, "missing_seq", "another_seq")
            .unwrap_err()
            .code,
        "42P01"
    );

    let mut catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    catalog.sequences.remove(old_sequence_name);
    catalog.sequences.remove(new_sequence_name);
    catalog.comments.remove(&CatalogCommentTarget::Sequence {
        sequence: old_sequence_name.to_string(),
    });
    catalog.comments.remove(&CatalogCommentTarget::Sequence {
        sequence: new_sequence_name.to_string(),
    });
}

#[test]
fn shared_catalog_persistence_carries_renamed_function_metadata() {
    let old_function_name = "shared_function_answer";
    let new_function_name = "shared_function_renamed_answer";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.functions.remove(old_function_name);
        catalog.functions.remove(new_function_name);
        catalog.comments.remove(&CatalogCommentTarget::Function {
            function: old_function_name.to_string(),
        });
        catalog.comments.remove(&CatalogCommentTarget::Function {
            function: new_function_name.to_string(),
        });
    }

    let mut session = Session::new(true);
    session.functions.insert(
        old_function_name.to_string(),
        FunctionInfo {
            oid: FIRST_USER_RELATION_OID,
            name: old_function_name.to_string(),
            return_type: SqlType::Int4,
            body: "SELECT 42".to_string(),
            acl: BTreeMap::new(),
        },
    );
    session.comments.insert(
        CatalogCommentTarget::Function {
            function: old_function_name.to_string(),
        },
        "metadata function".to_string(),
    );
    session.mark_function_dirty(old_function_name);
    session.mark_comment_dirty(CatalogCommentTarget::Function {
        function: old_function_name.to_string(),
    });
    session.persist_catalog_snapshot();

    rename_function_in_session(&mut session, old_function_name, new_function_name).unwrap();

    let reloaded = Session::new(true);
    assert!(pg_catalog_function_rows(&reloaded).contains(&vec![
        Some(FIRST_USER_RELATION_OID.to_string()),
        Some("public".to_string()),
        Some(new_function_name.to_string()),
        Some("23".to_string()),
        Some("integer".to_string()),
        Some("SELECT 42".to_string()),
    ]));
    assert!(
        psql_describe_function_verbose_rows(&reloaded).contains(&vec![
            Some("public".to_string()),
            Some(new_function_name.to_string()),
            Some("integer".to_string()),
            None,
            Some("func".to_string()),
            Some("volatile".to_string()),
            Some("unsafe".to_string()),
            Some("postgres".to_string()),
            Some("invoker".to_string()),
            None,
            Some("sql".to_string()),
            None,
            Some("metadata function".to_string()),
        ])
    );
    assert!(!pg_catalog_function_rows(&reloaded)
        .iter()
        .any(|row| row.get(2).and_then(Option::as_deref) == Some(old_function_name)));
    assert_eq!(
        rename_function_in_session(&mut session, "missing_function", "another_function")
            .unwrap_err()
            .code,
        "42883"
    );

    let mut catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    catalog.functions.remove(old_function_name);
    catalog.functions.remove(new_function_name);
    catalog.comments.remove(&CatalogCommentTarget::Function {
        function: old_function_name.to_string(),
    });
    catalog.comments.remove(&CatalogCommentTarget::Function {
        function: new_function_name.to_string(),
    });
}

#[test]
fn shared_catalog_persistence_carries_renamed_table_metadata() {
    let old_table_name = "shared_rename_table_people";
    let new_table_name = "shared_renamed_table_people";
    let index_name = "shared_rename_table_people_id_idx";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tables.remove(old_table_name);
        catalog.tables.remove(new_table_name);
        catalog.indexes.retain(|index| index.name != index_name);
        catalog.comments.remove(&CatalogCommentTarget::Table {
            table: old_table_name.to_string(),
        });
        catalog.comments.remove(&CatalogCommentTarget::Table {
            table: new_table_name.to_string(),
        });
        catalog.comments.remove(&CatalogCommentTarget::Constraint {
            table: old_table_name.to_string(),
            constraint: index_name.to_string(),
        });
        catalog.comments.remove(&CatalogCommentTarget::Constraint {
            table: new_table_name.to_string(),
            constraint: index_name.to_string(),
        });
    }

    let mut session = Session::new(true);
    session.tables.insert(
        old_table_name.to_string(),
        test_table(old_table_name, vec![vec![SqlValue::Int4(1)]]),
    );
    session.indexes.push(CatalogIndex {
        name: index_name.to_string(),
        table: old_table_name.to_string(),
        column: "id".to_string(),
        unique: true,
        primary_key: true,
        unique_constraint: false,
    });
    session.comments.insert(
        CatalogCommentTarget::Table {
            table: old_table_name.to_string(),
        },
        "shared table".to_string(),
    );
    session.comments.insert(
        CatalogCommentTarget::Constraint {
            table: old_table_name.to_string(),
            constraint: index_name.to_string(),
        },
        "shared pkey".to_string(),
    );
    session.mark_table_dirty(old_table_name);
    session.dirty_indexes = true;
    session.persist_catalog_snapshot();

    rename_table_in_session(&mut session, old_table_name, new_table_name, false).unwrap();

    let reloaded = Session::new(true);
    assert!(!reloaded.tables.contains_key(old_table_name));
    assert!(reloaded.tables.contains_key(new_table_name));
    assert!(pg_catalog_index_rows(&reloaded).contains(&vec![
        Some("public".to_string()),
        Some(new_table_name.to_string()),
        Some(index_name.to_string()),
        Some(format!(
            "CREATE UNIQUE INDEX {index_name} ON public.{new_table_name} USING btree (id)"
        )),
    ]));
    assert!(pg_catalog_table_description_rows(&reloaded).contains(&vec![
        Some("public".to_string()),
        Some(new_table_name.to_string()),
        None,
        Some("shared table".to_string()),
    ]));
    assert!(
        pg_catalog_constraint_description_rows(&reloaded).contains(&vec![
            Some("public".to_string()),
            Some(new_table_name.to_string()),
            Some(index_name.to_string()),
            Some("shared pkey".to_string()),
        ])
    );
    assert_eq!(
        rename_table_in_session(&mut session, "missing_shared_table", "unused", false)
            .unwrap_err()
            .code,
        "42P01"
    );
    rename_table_in_session(&mut session, "missing_shared_table", "unused", true).unwrap();

    let mut catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    catalog.tables.remove(old_table_name);
    catalog.tables.remove(new_table_name);
    catalog.indexes.retain(|index| index.name != index_name);
    catalog.comments.remove(&CatalogCommentTarget::Table {
        table: old_table_name.to_string(),
    });
    catalog.comments.remove(&CatalogCommentTarget::Table {
        table: new_table_name.to_string(),
    });
    catalog.comments.remove(&CatalogCommentTarget::Constraint {
        table: old_table_name.to_string(),
        constraint: index_name.to_string(),
    });
    catalog.comments.remove(&CatalogCommentTarget::Constraint {
        table: new_table_name.to_string(),
        constraint: index_name.to_string(),
    });
}

#[test]
fn shared_catalog_persistence_merges_dirty_tables() {
    let accounts = "parallel_restore_accounts";
    let events = "parallel_restore_events";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tables.remove(accounts);
        catalog.tables.remove(events);
    }

    let mut accounts_session = Session::new(true);
    accounts_session.tables.insert(
        accounts.to_string(),
        test_table(accounts, vec![vec![SqlValue::Int4(1)]]),
    );
    accounts_session.tables.insert(
        events.to_string(),
        test_table(events, vec![vec![SqlValue::Int4(10)]]),
    );

    let mut events_session = Session::new(true);
    events_session.tables = accounts_session.tables.clone();
    accounts_session
        .tables
        .get_mut(accounts)
        .unwrap()
        .rows
        .push(vec![SqlValue::Int4(2)]);
    accounts_session.mark_table_dirty(accounts);
    accounts_session.persist_catalog_snapshot();

    events_session
        .tables
        .get_mut(events)
        .unwrap()
        .rows
        .push(vec![SqlValue::Int4(11)]);
    events_session.mark_table_dirty(events);
    events_session.persist_catalog_snapshot();

    let catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    assert_eq!(
        catalog.tables[accounts].rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(2)]]
    );
    assert_eq!(
        catalog.tables[events].rows,
        vec![vec![SqlValue::Int4(10)], vec![SqlValue::Int4(11)]]
    );
}

#[test]
fn shared_catalog_index_update_does_not_overwrite_table_rows() {
    let table_name = "parallel_index_restore_accounts";
    let index_name = "parallel_index_restore_accounts_id_idx";
    let other_table_name = "parallel_index_restore_events";
    let other_index_name = "parallel_index_restore_events_id_idx";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tables.remove(table_name);
        catalog.tables.remove(other_table_name);
        catalog.indexes.retain(|index| index.name != index_name);
        catalog
            .indexes
            .retain(|index| index.name != other_index_name);
    }

    let mut data_session = Session::new(true);
    data_session.tables.insert(
        table_name.to_string(),
        test_table(table_name, vec![vec![SqlValue::Int4(1)]]),
    );
    data_session.tables.insert(
        other_table_name.to_string(),
        test_table(other_table_name, vec![vec![SqlValue::Int4(10)]]),
    );
    data_session.mark_table_dirty(table_name);
    data_session.mark_table_dirty(other_table_name);
    data_session.persist_catalog_snapshot();

    let mut index_session = Session::new(true);
    let mut other_index_session = Session::new(true);
    index_session.indexes.push(CatalogIndex {
        name: index_name.to_string(),
        table: table_name.to_string(),
        column: "id".to_string(),
        unique: false,
        primary_key: false,
        unique_constraint: false,
    });
    index_session.dirty_indexes = true;
    other_index_session.indexes.push(CatalogIndex {
        name: other_index_name.to_string(),
        table: other_table_name.to_string(),
        column: "id".to_string(),
        unique: false,
        primary_key: false,
        unique_constraint: false,
    });
    other_index_session.dirty_indexes = true;

    data_session
        .tables
        .get_mut(table_name)
        .unwrap()
        .rows
        .push(vec![SqlValue::Int4(2)]);
    data_session.mark_table_dirty(table_name);
    data_session.persist_catalog_snapshot();
    other_index_session.persist_catalog_snapshot();
    index_session.persist_catalog_snapshot();

    let catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    assert_eq!(
        catalog.tables[table_name].rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(2)]]
    );
    assert!(catalog
        .indexes
        .iter()
        .any(|index| index.name == index_name && index.table == table_name));
    assert!(catalog
        .indexes
        .iter()
        .any(|index| index.name == other_index_name && index.table == other_table_name));
    drop(catalog);

    let mut catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    catalog.tables.remove(table_name);
    catalog.tables.remove(other_table_name);
    catalog.indexes.retain(|index| index.name != index_name);
    catalog
        .indexes
        .retain(|index| index.name != other_index_name);
}

#[test]
fn shared_catalog_persistence_removes_dirty_deleted_tables() {
    let table = "clean_restore_accounts";
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tables.insert(
            table.to_string(),
            test_table(table, vec![vec![SqlValue::Int4(99)]]),
        );
    }

    let mut session = Session::new(true);
    assert!(session.tables.remove(table).is_some());
    session.mark_table_dirty(table);
    session.persist_catalog_snapshot();

    let catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    assert!(!catalog.tables.contains_key(table));
}

#[test]
fn shared_catalog_table_drop_removes_indexes_and_comments() {
    let table = "drop_shared_accounts";
    let index = "drop_shared_accounts_id_idx";
    let table_target = CatalogCommentTarget::Table {
        table: table.to_string(),
    };
    let column_target = CatalogCommentTarget::Column {
        table: table.to_string(),
        attnum: 1,
    };
    let index_target = CatalogCommentTarget::Index {
        index: index.to_string(),
    };
    let schema_target = CatalogCommentTarget::Schema {
        schema: "public".to_string(),
    };
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.tables.insert(
            table.to_string(),
            test_table(table, vec![vec![SqlValue::Int4(99)]]),
        );
        catalog.indexes.push(CatalogIndex {
            name: index.to_string(),
            table: table.to_string(),
            column: "id".to_string(),
            unique: false,
            primary_key: false,
            unique_constraint: false,
        });
        catalog
            .comments
            .insert(table_target.clone(), "table comment".to_string());
        catalog
            .comments
            .insert(column_target.clone(), "column comment".to_string());
        catalog
            .comments
            .insert(index_target.clone(), "index comment".to_string());
        catalog
            .comments
            .insert(schema_target.clone(), "schema comment".to_string());
    }

    let mut session = Session::new(true);
    assert!(session.tables.remove(table).is_some());
    session.indexes.retain(|candidate| candidate.table != table);
    session.dirty_indexes = true;
    for target in [&table_target, &column_target, &index_target] {
        session.comments.remove(target);
        session.mark_comment_dirty(target.clone());
    }
    session.mark_table_dirty(table);
    session.persist_catalog_snapshot();

    let catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    assert!(!catalog.tables.contains_key(table));
    assert!(!catalog
        .indexes
        .iter()
        .any(|candidate| candidate.table == table));
    assert!(!catalog.comments.contains_key(&table_target));
    assert!(!catalog.comments.contains_key(&column_target));
    assert!(!catalog.comments.contains_key(&index_target));
    assert_eq!(
        catalog.comments.get(&schema_target),
        Some(&"schema comment".to_string())
    );
}

#[test]
fn shared_catalog_persistence_removes_dirty_deleted_views() {
    let view = "clean_restore_active_people";
    let Command::Select(query) = parse_command("SELECT id, name FROM people ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    {
        let mut catalog = shared_catalog()
            .lock()
            .expect("shared catalog mutex poisoned");
        catalog.views.insert(
            view.to_string(),
            View {
                oid: FIRST_USER_RELATION_OID,
                name: view.to_string(),
                query,
                definition: "SELECT id, name FROM people ORDER BY id".to_string(),
            },
        );
    }

    let mut session = Session::new(true);
    assert!(session.views.remove(view).is_some());
    session.mark_view_dirty(view);
    session.persist_catalog_snapshot();

    let catalog = shared_catalog()
        .lock()
        .expect("shared catalog mutex poisoned");
    assert!(!catalog.views.contains_key(view));
}
