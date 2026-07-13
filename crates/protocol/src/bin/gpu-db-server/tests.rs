use super::*;
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

fn tcp_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let client = TcpStream::connect(addr).unwrap();
    let (server, _) = listener.accept().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    (server, client)
}

fn read_backend_messages(stream: &mut dyn ReadWrite, count: usize) -> Vec<(u8, Vec<u8>)> {
    let mut messages = Vec::with_capacity(count);
    for _ in 0..count {
        let mut tag = [0_u8; 1];
        stream.read_exact(&mut tag).unwrap();
        let mut len = [0_u8; 4];
        stream.read_exact(&mut len).unwrap();
        let payload_len = u32::from_be_bytes(len) as usize - 4;
        let mut payload = vec![0_u8; payload_len];
        stream.read_exact(&mut payload).unwrap();
        messages.push((tag[0], payload));
    }
    messages
}

fn read_backend_tags(stream: &mut dyn ReadWrite, count: usize) -> Vec<u8> {
    let messages = read_backend_messages(stream, count);
    let mut tags = Vec::with_capacity(messages.len());
    for (tag, _) in messages {
        tags.push(tag);
    }
    tags
}

#[test]
fn asyncpg_default_pool_session_reset_query_is_session_control_noop() {
    let mut session = Session::default();
    let (mut writer, mut reader) = tcp_pair();

    run_simple_query(
        &mut writer,
        &mut session,
        "SELECT pg_advisory_unlock_all(); CLOSE ALL; UNLISTEN *; RESET ALL;",
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 7);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'C', b'C', b'C', b'Z']
    );
    assert_eq!(messages[2].1, b"SELECT 1\0".to_vec());

    run_simple_query(&mut writer, &mut session, "SELECT 1 AS one;").unwrap();
    let messages = read_backend_messages(&mut reader, 4);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'Z']
    );
    assert_eq!(messages[2].1, b"SELECT 1\0".to_vec());
}

fn error_field_value(payload: &[u8], field_tag: u8) -> Option<String> {
    let mut idx = 0;
    while idx < payload.len() {
        let tag = payload[idx];
        idx += 1;
        if tag == 0 {
            break;
        }
        let end = payload[idx..]
            .iter()
            .position(|byte| *byte == 0)
            .map(|offset| idx + offset)?;
        let value = std::str::from_utf8(&payload[idx..end]).ok()?;
        if tag == field_tag {
            return Some(value.to_string());
        }
        idx = end + 1;
    }
    None
}

fn test_table(name: &str, rows: Vec<Vec<SqlValue>>) -> Table {
    Table {
        oid: FIRST_USER_RELATION_OID,
        name: name.to_string(),
        columns: vec![CatalogColumn {
            attnum: 1,
            def: gpu_db_protocol::ColumnDef {
                name: "id".to_string(),
                ty: SqlType::Int4,
                domain: None,
                default: None,
            },
        }],
        rows,
        check_constraints: Vec::new(),
        foreign_keys: Vec::new(),
    }
}

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

#[test]
fn canonical_sql_collapses_case_whitespace_and_semicolons() {
    assert_eq!(
        canonical_sql("  SELECT   1   AS One ; ; "),
        "select 1 as one"
    );
}

#[test]
fn recognizes_pg_dumpall_tablespace_metadata_query() {
    let canonical = canonical_sql(
        "SELECT oid, spcname, pg_catalog.pg_get_userbyid(spcowner) AS spcowner, pg_catalog.pg_tablespace_location(oid), spcacl, acldefault('t', spcowner) AS acldefault, array_to_string(spcoptions, ', '),pg_catalog.shobj_description(oid, 'pg_tablespace') FROM pg_catalog.pg_tablespace WHERE spcname !~ '^pg_' ORDER BY 1",
    );
    assert!(is_pg_dumpall_tablespace_metadata_query(&canonical));
}

#[test]
fn copy_statement_detection_skips_leading_comments() {
    assert!(is_copy_statement(
        "/* copy boundary */ -- line comment\nCOPY copy_people TO STDOUT;"
    ));
    assert!(is_copy_statement(
        "/* outer /* nested */ done */ COPY copy_people FROM STDIN;"
    ));
    assert!(!is_copy_statement(
        "/* copy-looking comment */ SELECT 'COPY people TO STDOUT'"
    ));
    assert!(!is_copy_statement(
        "/* unterminated COPY copy_people TO STDOUT"
    ));
}

#[test]
fn copy_to_stdout_table_detection_is_narrow() {
    assert_eq!(
        parse_copy_to_stdout_table("COPY public.people TO STDOUT;"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions::TEXT,
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table("/* comment */ COPY people TO STDOUT"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions::TEXT,
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table("COPY people TO STDOUT WITH CSV"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions::CSV,
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table("COPY people TO STDOUT WITH CSV HEADER"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions::CSV_HEADER,
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table(
            "COPY people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|')"
        ),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: true,
                delimiter: '|',
                quote: '"',
                escape: '"',
            }
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table(
            "COPY people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|', QUOTE '''', ESCAPE '\\')"
        ),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: true,
                delimiter: '|',
                quote: '\'',
                escape: '\\',
            }
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table("COPY people TO STDOUT WITH (FORMAT csv, QUOTE '|')"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: false,
                delimiter: ',',
                quote: '|',
                escape: '|',
            }
        })
    );
    assert_eq!(parse_copy_to_stdout_table("COPY people FROM STDIN"), None);
    assert_eq!(
        parse_copy_to_stdout_table("COPY (SELECT * FROM people) TO STDOUT"),
        None
    );
    assert_eq!(
        parse_copy_to_stdout_table("COPY people TO STDOUT WITH (FORMAT csv, NULL '')"),
        None
    );
}

#[test]
fn copy_from_stdin_table_detection_is_narrow() {
    assert_eq!(
        parse_copy_from_stdin("COPY public.people FROM STDIN;"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: None,
            options: CopyOptions::TEXT,
        })
    );
    assert_eq!(
        parse_copy_from_stdin("/* comment */ COPY people (id, name) FROM STDIN"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions::TEXT,
        })
    );
    assert_eq!(
        parse_copy_from_stdin("COPY people FROM STDIN WITH CSV"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: None,
            options: CopyOptions::CSV,
        })
    );
    assert_eq!(
        parse_copy_from_stdin("COPY public.people (id, name) FROM STDIN WITH CSV"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions::CSV,
        })
    );
    assert_eq!(
        parse_copy_from_stdin("COPY public.people (id, name) FROM STDIN WITH CSV HEADER"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions::CSV_HEADER,
        })
    );
    assert_eq!(
        parse_copy_from_stdin(
            "COPY public.people (id, name) FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER '|')"
        ),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: true,
                delimiter: '|',
                quote: '"',
                escape: '"',
            }
        })
    );
    assert_eq!(
        parse_copy_from_stdin(
            "COPY public.people (id, name) FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER '|', QUOTE '''', ESCAPE '\\')"
        ),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: true,
                delimiter: '|',
                quote: '\'',
                escape: '\\',
            }
        })
    );
    assert_eq!(parse_copy_from_stdin("COPY people TO STDOUT"), None);
    assert_eq!(
        parse_copy_from_stdin("COPY (SELECT * FROM people) FROM STDIN"),
        None
    );
    assert_eq!(
        parse_copy_from_stdin("COPY people FROM STDIN WITH (FORMAT csv, DELIMITER '|', QUOTE '|')"),
        None
    );
}

#[test]
fn truncate_table_detection_is_narrow() {
    assert_eq!(
        parse_truncate_table("TRUNCATE TABLE ONLY public.people;"),
        Some(ParsedTruncateTable {
            table: "people".to_string(),
            restart_identity: false,
        })
    );
    assert_eq!(
        parse_truncate_table("/* restore */ TRUNCATE TABLE people;"),
        Some(ParsedTruncateTable {
            table: "people".to_string(),
            restart_identity: false,
        })
    );
    assert_eq!(
        parse_truncate_table("TRUNCATE people"),
        Some(ParsedTruncateTable {
            table: "people".to_string(),
            restart_identity: false,
        })
    );
    assert_eq!(
        parse_truncate_table("TRUNCATE TABLE public.people RESTART IDENTITY"),
        Some(ParsedTruncateTable {
            table: "people".to_string(),
            restart_identity: true,
        })
    );
    assert_eq!(
        parse_truncate_table("TRUNCATE TABLE public.people CONTINUE IDENTITY"),
        None
    );
    assert_eq!(parse_truncate_table("TRUNCATE TABLE people CASCADE"), None);
    assert_eq!(parse_truncate_table("TRUNCATE TABLE people, teams"), None);
    assert_eq!(parse_truncate_table("TRUNCATE TABLE private.people"), None);
    assert_eq!(
        parse_truncate_table("TRUNCATE TABLE public.people CASCADE"),
        None
    );
    assert_eq!(parse_truncate_table("TRUNCATE TABLE \"people\""), None);
}

#[test]
fn drop_table_detection_is_narrow() {
    assert_eq!(
        parse_drop_table("DROP TABLE IF EXISTS public.people;"),
        Some(DropTable {
            tables: vec!["people".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_drop_table("/* restore */ DROP TABLE people;"),
        Some(DropTable {
            tables: vec!["people".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(
        parse_drop_table("DROP TABLE public.people, teams"),
        Some(DropTable {
            tables: vec!["people".to_string(), "teams".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(parse_drop_table("DROP TABLE public.people CASCADE"), None);
    assert_eq!(parse_drop_table("DROP SCHEMA IF EXISTS public"), None);
    assert_eq!(parse_drop_table("DROP TABLE \"people\""), None);
    assert_eq!(parse_drop_table("DROP TABLE private.people, teams"), None);
    assert_eq!(
        parse_alter_table_drop_constraint(
            "ALTER TABLE IF EXISTS ONLY public.accounts DROP CONSTRAINT IF EXISTS accounts_pkey;"
        ),
        Some(DropConstraint {
            table: "accounts".to_string(),
            constraint: "accounts_pkey".to_string(),
            table_if_exists: true,
            if_exists: true,
        })
    );
    assert_eq!(
        parse_alter_table_drop_constraint(
            "ALTER TABLE ONLY accounts DROP CONSTRAINT accounts_pkey;"
        ),
        Some(DropConstraint {
            table: "accounts".to_string(),
            constraint: "accounts_pkey".to_string(),
            table_if_exists: false,
            if_exists: false,
        })
    );
    assert_eq!(
        parse_alter_table_drop_constraint("ALTER TABLE accounts DROP COLUMN id"),
        None
    );
}

#[test]
fn simple_copy_to_stdout_emits_copyout_data_done_and_recovers() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Tab\tName".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(&mut writer, &mut session, "COPY people TO STDOUT", true).unwrap();

    let messages = read_backend_messages(&mut reader, 5);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[0].1, vec![0, 0, 2, 0, 0, 0, 0]);
    assert_eq!(messages[1].1, b"1\tAda\n");
    assert_eq!(messages[2].1, b"2\tTab\\tName\n");
    assert_eq!(messages[4].1, b"COPY 2\0");

    execute_statement(
        &mut writer,
        &mut session,
        "SELECT name FROM people WHERE id = 1",
        true,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_copy_to_stdout_with_csv_quotes_fields_and_recovers() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![
                    SqlValue::Int4(2),
                    SqlValue::Text("Grace, \"Amazing\"".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "COPY people TO STDOUT WITH CSV",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 5);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[1].1, b"1,Ada\n");
    assert_eq!(messages[2].1, b"2,\"Grace, \"\"Amazing\"\"\"\n");
    assert_eq!(messages[4].1, b"COPY 2\0");

    execute_statement(
        &mut writer,
        &mut session,
        "SELECT name FROM people WHERE id = 2",
        true,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_copy_to_stdout_with_csv_header_emits_column_names_and_recovers() {
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
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "full_name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "COPY people TO STDOUT WITH CSV HEADER",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 5);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[1].1, b"id,full_name\n");
    assert_eq!(messages[2].1, b"1,Ada\n");
    assert_eq!(messages[4].1, b"COPY 1\0");

    execute_statement(
        &mut writer,
        &mut session,
        "SELECT full_name FROM people WHERE id = 1",
        true,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_copy_to_stdout_with_parenthesized_csv_delimiter_recovers() {
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
                        ty: SqlType::Int4,
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "full_name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![
                    SqlValue::Int4(2),
                    SqlValue::Text("Grace|Hopper".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "COPY people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|')",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 6);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[1].1, b"id|full_name\n");
    assert_eq!(messages[2].1, b"1|Ada\n");
    assert_eq!(messages[3].1, b"2|\"Grace|Hopper\"\n");
    assert_eq!(messages[5].1, b"COPY 2\0");

    execute_statement(
        &mut writer,
        &mut session,
        "SELECT full_name FROM people WHERE id = 2",
        true,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_copy_from_stdin_accepts_data_done_and_recovers() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("COPY people FROM STDIN".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'G');
    assert_eq!(messages[0].1, vec![0, 0, 2, 0, 0, 0, 0]);
    assert!(session.copy_in.is_some());

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"1\tAda\n2\tGrace\\tHopper\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace\tHopper".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn simple_copy_from_stdin_with_parenthesized_csv_delimiter_recovers() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery(
            "COPY people FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER '|')".to_string()
        )
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'G');

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"id|name\n1|Ada\n2|\"Grace|Hopper\"\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace|Hopper".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn simple_copy_from_stdin_with_csv_accepts_quoted_data_and_recovers() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("COPY people FROM STDIN WITH CSV".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'G');
    assert_eq!(messages[0].1, vec![0, 0, 2, 0, 0, 0, 0]);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"1,Ada\n2,\"Grace, \"\"Hopper\"\"\"\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace, \"Hopper\"".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn simple_copy_from_stdin_with_csv_header_skips_header_and_recovers() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("COPY people FROM STDIN WITH CSV HEADER".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'G');

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"id,name\n1,Ada\n2,\"Grace, \"\"Hopper\"\"\"\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace, \"Hopper\"".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn extended_copy_to_stdout_emits_copyout_data_done_and_recovers() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![
                    SqlValue::Int4(2),
                    SqlValue::Text("Grace|Hopper".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "copy_out".to_string(),
            query: "COPY people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|')".to_string(),
            parameter_type_oids: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "copy_out_portal".to_string(),
            statement_name: "copy_out".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "copy_out_portal".to_string(),
        }
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'1', b'2', b'n']);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "copy_out_portal".to_string(),
            max_rows: 0,
        }
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 6);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[1].1, b"id|name\n");
    assert_eq!(messages[2].1, b"1|Ada\n");
    assert_eq!(messages[3].1, b"2|\"Grace|Hopper\"\n");
    assert_eq!(messages[5].1, b"COPY 2\0");

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn extended_copy_from_stdin_accepts_data_and_copyfail_does_not_mutate() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "copy_in".to_string(),
            query: "COPY people (id, name) FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER '|')"
                .to_string(),
            parameter_type_oids: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "copy_in_portal".to_string(),
            statement_name: "copy_in".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "copy_in_portal".to_string(),
            max_rows: 0,
        }
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 3);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'1', b'2', b'G']
    );
    assert!(session.copy_in.is_some());

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"id|name\n1|Ada\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyFail("client aborted copy".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E']
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("57014".to_string())
    );
    assert!(session.tables["people"].rows.is_empty());
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "copy_in_retry".to_string(),
            query: "COPY people (id, name) FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER '|')"
                .to_string(),
            parameter_type_oids: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "copy_in_retry_portal".to_string(),
            statement_name: "copy_in_retry".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "copy_in_retry_portal".to_string(),
            max_rows: 0,
        }
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'1', b'2', b'G']);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"id|name\n1|Ada\n2|\"Grace|Hopper\"\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace|Hopper".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn truncate_table_clears_rows_and_recovers() {
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
                    ty: SqlType::Int4,
                    domain: None,
                    default: Some(ColumnDefault::SequenceNextVal {
                        sequence: "people_id_seq".to_string(),
                        create_if_missing: false,
                    }),
                },
            }],
            rows: vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(2)]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.sequences.insert(
        "people_id_seq".to_string(),
        Sequence {
            oid: FIRST_USER_RELATION_OID + 1,
            name: "people_id_seq".to_string(),
            last_value: 2,
            is_called: true,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "TRUNCATE TABLE ONLY public.people RESTART IDENTITY",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"TRUNCATE TABLE\0");
    assert!(session.tables["people"].rows.is_empty());
    assert_eq!(session.sequences["people_id_seq"].last_value, 1);
    assert!(!session.sequences["people_id_seq"].is_called);

    execute_statement(&mut writer, &mut session, "SELECT id FROM people", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'T', b'C']);
}

#[test]
fn simple_relational_delete_removes_matching_rows_and_recovers() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
                vec![
                    SqlValue::Int4(4),
                    SqlValue::Text("Ada Lovelace".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "DELETE FROM people WHERE id = 2 OR name LIKE 'Ada%'",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"DELETE 3\0");
    assert_eq!(
        session.tables["people"].rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())]]
    );

    execute_statement(&mut writer, &mut session, "SELECT id FROM people", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_relational_update_changes_matching_rows_and_recovers() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
                vec![
                    SqlValue::Int4(4),
                    SqlValue::Text("Ada Lovelace".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "UPDATE people SET name = 'Updated' WHERE id = 2 OR name LIKE 'Ada%'",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"UPDATE 3\0");
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Updated".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Updated".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            vec![SqlValue::Int4(4), SqlValue::Text("Updated".to_string())],
        ]
    );

    execute_statement(&mut writer, &mut session, "SELECT id FROM people", true).unwrap();
    assert_eq!(
        read_backend_tags(&mut reader, 6),
        vec![b'T', b'D', b'D', b'D', b'D', b'C']
    );
}

#[test]
fn split_simple_query_discards_empty_segments() {
    assert_eq!(
        split_simple_query("SELECT 1;;  SELECT 2;"),
        vec!["SELECT 1", "SELECT 2"]
    );
}

#[test]
fn split_simple_query_preserves_semicolons_inside_text_literals() {
    assert_eq!(
        split_simple_query(
            r#"INSERT INTO commands VALUES ('SELECT 1;'); PREPARE "lookup;name"(int4) AS SELECT 'Ada'';Lovelace';"#
        ),
        vec![
            "INSERT INTO commands VALUES ('SELECT 1;')",
            r#"PREPARE "lookup;name"(int4) AS SELECT 'Ada'';Lovelace'"#
        ]
    );
}

#[test]
fn split_simple_query_preserves_semicolons_inside_sql_comments() {
    assert_eq!(
        split_simple_query(
            "/* comment ; /* nested ; */ done */ PREPARE lookup(int4) AS SELECT id FROM people WHERE id = $1; \
             -- comment ; before execute\n\
             EXECUTE lookup(1);"
        ),
        vec![
            "/* comment ; /* nested ; */ done */ PREPARE lookup(int4) AS SELECT id FROM people WHERE id = $1",
            "-- comment ; before execute\nEXECUTE lookup(1)"
        ]
    );
}

#[test]
fn frontend_function_call_unsupported_error_is_explicit() {
    assert_eq!(
        unsupported_frontend_message(&FrontendMessage::FunctionCall {
            function_oid: 42,
            argument_format_codes: vec![],
            arguments: vec![],
            result_format_code: 0,
        }),
        "FunctionCall is not supported by the compatibility endpoint"
    );
    assert_eq!(
        unsupported_frontend_message(&FrontendMessage::CopyData(Vec::new())),
        "frontend COPY data flow is not supported by the compatibility endpoint"
    );
    assert_eq!(
        unsupported_frontend_message(&FrontendMessage::CopyDone),
        "frontend COPY data flow is not supported by the compatibility endpoint"
    );
    assert_eq!(
        unsupported_frontend_message(&FrontendMessage::CopyFail(
            "client aborted copy".to_string()
        )),
        "frontend COPY data flow is not supported by the compatibility endpoint"
    );
}

#[test]
fn frontend_function_call_error_skips_until_sync_and_recovers() {
    let mut session = Session::default();
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::FunctionCall {
            function_oid: 42,
            argument_format_codes: vec![],
            arguments: vec![],
            result_format_code: 0,
        }
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("0A000".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("FunctionCall is not supported by the compatibility endpoint".to_string())
    );
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_function_call (id INT)".to_string())
    )
    .unwrap());
    assert!(!session.tables.contains_key("skipped_function_call"));

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE recovered_function_call (id INT)".to_string())
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert!(session.tables.contains_key("recovered_function_call"));
}

#[test]
fn frontend_copy_data_error_skips_until_sync_and_recovers() {
    let mut session = Session::default();
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"1\tAda\n".to_vec())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("0A000".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("frontend COPY data flow is not supported by the compatibility endpoint".to_string())
    );
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_copy_frame (id INT)".to_string())
    )
    .unwrap());
    assert!(!session.tables.contains_key("skipped_copy_frame"));

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE recovered_copy_frame (id INT)".to_string())
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert!(session.tables.contains_key("recovered_copy_frame"));
}

#[test]
fn frontend_copy_done_error_skips_until_sync_and_recovers() {
    let mut session = Session::default();
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("0A000".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("frontend COPY data flow is not supported by the compatibility endpoint".to_string())
    );
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_copy_done (id INT)".to_string())
    )
    .unwrap());
    assert!(!session.tables.contains_key("skipped_copy_done"));

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE recovered_copy_done (id INT)".to_string())
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert!(session.tables.contains_key("recovered_copy_done"));
}

#[test]
fn frontend_copy_fail_error_skips_until_sync_and_recovers() {
    let mut session = Session::default();
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyFail("client aborted copy".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("0A000".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("frontend COPY data flow is not supported by the compatibility endpoint".to_string())
    );
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_copy_fail (id INT)".to_string())
    )
    .unwrap());
    assert!(!session.tables.contains_key("skipped_copy_fail"));

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE recovered_copy_fail (id INT)".to_string())
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert!(session.tables.contains_key("recovered_copy_fail"));
}

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

#[test]
fn extended_parameter_binding_substitutes_text_and_int_literals() {
    let query = PreparedQuery {
        query: "SELECT id, name FROM people WHERE id = $1 ORDER BY name LIMIT $2".to_string(),
        parameter_type_oids: vec![23, 23],
    };

    assert_eq!(
        bind_query_parameters(&query, &[Some("2".to_string()), Some("1".to_string())]),
        Ok("SELECT id, name FROM people WHERE id = 2 ORDER BY name LIMIT 1".to_string())
    );

    let text_query = PreparedQuery {
        query: "SELECT id FROM people WHERE name = $1".to_string(),
        parameter_type_oids: vec![25],
    };
    assert_eq!(
        bind_query_parameters(&text_query, &[Some("O'Brien".to_string())]),
        Ok("SELECT id FROM people WHERE name = 'O''Brien'".to_string())
    );
    assert_eq!(
        bind_query_parameters(&text_query, &[Some("Ada $1".to_string())]),
        Ok("SELECT id FROM people WHERE name = 'Ada $1'".to_string())
    );
}

#[test]
fn extended_parameter_binding_replaces_exact_placeholder_tokens() {
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1 OR id = $10 ORDER BY id LIMIT $11".to_string(),
        parameter_type_oids: vec![23; 11],
    };
    let parameters = (1..=11)
        .map(|idx| Some(idx.to_string()))
        .collect::<Vec<_>>();

    assert_eq!(
        replace_unquoted_placeholder(&query.query, "$1", "7"),
        "SELECT id FROM people WHERE id = 7 OR id = $10 ORDER BY id LIMIT $11"
    );
    assert_eq!(
        bind_query_parameters(&query, &parameters),
        Ok("SELECT id FROM people WHERE id = 1 OR id = 10 ORDER BY id LIMIT 11".to_string())
    );
}

#[test]
fn extended_parameter_binding_matches_zero_padded_nonzero_placeholders() {
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $01 OR id = $002 ORDER BY id LIMIT $0003"
            .to_string(),
        parameter_type_oids: vec![23, 23, 23],
    };

    assert_eq!(max_placeholder_index(&query.query), 3);
    assert!(!contains_zero_placeholder(&query.query));
    assert_eq!(
        replace_unquoted_placeholder(&query.query, "$1", "7"),
        "SELECT id FROM people WHERE id = 7 OR id = $002 ORDER BY id LIMIT $0003"
    );
    assert_eq!(
        bind_query_parameters(
            &query,
            &[
                Some("1".to_string()),
                Some("2".to_string()),
                Some("2".to_string()),
            ],
        ),
        Ok("SELECT id FROM people WHERE id = 1 OR id = 2 ORDER BY id LIMIT 2".to_string())
    );
}

#[test]
fn extended_parameter_binding_ignores_quoted_placeholder_literals() {
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE name = '$1' OR name = 'O''$2' OR id = $1".to_string(),
        parameter_type_oids: vec![23],
    };

    assert_eq!(max_placeholder_index(&query.query), 1);
    assert_eq!(expected_parameter_count(&query), 1);
    assert_eq!(
        bind_query_parameters(&query, &[Some("2".to_string())]),
        Ok("SELECT id FROM people WHERE name = '$1' OR name = 'O''$2' OR id = 2".to_string())
    );
    assert_eq!(
        replace_parameter_placeholders_with_dummy_literals(
            "SELECT id FROM people WHERE name = '$1' OR id = $1"
        ),
        "SELECT id FROM people WHERE name = '$1' OR id = 1"
    );
}

#[test]
fn extended_parameter_binding_ignores_commented_placeholder_literals() {
    let query = strip_sql_comments(
        "SELECT id FROM people WHERE id = $1 -- ignored $2\n\
         ORDER BY id /* ignored $3 */ LIMIT $2",
    );
    let prepared = PreparedQuery {
        query,
        parameter_type_oids: vec![23, 23],
    };

    assert_eq!(max_placeholder_index(&prepared.query), 2);
    assert_eq!(expected_parameter_count(&prepared), 2);
    assert_eq!(
        bind_query_parameters(&prepared, &[Some("2".to_string()), Some("1".to_string())]),
        Ok("SELECT id FROM people WHERE id = 2 \nORDER BY id   LIMIT 1".to_string())
    );
    assert_eq!(
        strip_sql_comments("SELECT '$1 -- still text', id FROM people WHERE id = $1"),
        "SELECT '$1 -- still text', id FROM people WHERE id = $1"
    );
    assert_eq!(
        strip_sql_comments(
            "SELECT $$-- still text$$, $tag$/* still text */$tag$, id FROM people WHERE id = $1"
        ),
        "SELECT $$-- still text$$, $tag$/* still text */$tag$, id FROM people WHERE id = $1"
    );
    assert_eq!(
        strip_sql_comments(r#"SELECT "-- still identifier", id FROM people WHERE id = $1"#),
        r#"SELECT "-- still identifier", id FROM people WHERE id = $1"#
    );
    assert_eq!(
        strip_sql_comments(
            "SELECT id FROM people WHERE id = $1 /* outer $2 /* inner $3 */ done $4 */ LIMIT $2"
        ),
        "SELECT id FROM people WHERE id = $1   LIMIT $2"
    );
}

#[test]
fn extended_parse_rejects_zero_placeholder_without_installing_statement() {
    let mut session = Session::default();
    assert!(contains_zero_placeholder(
        "SELECT name FROM people WHERE id = $0"
    ));
    assert!(contains_zero_placeholder(
        "SELECT name FROM people WHERE id = $00"
    ));
    assert!(!contains_zero_placeholder(
        "SELECT '$0' AS literal, name FROM people WHERE id = $1"
    ));
    assert!(!contains_zero_placeholder(
        "SELECT name FROM people WHERE id = $01"
    ));
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
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "bad_zero".to_string(),
            query: "SELECT name FROM people WHERE id = $0".to_string(),
            parameter_type_oids: Vec::new(),
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("42P02".to_string())
    );
    assert!(!session.prepared.contains_key("bad_zero"));
    assert!(extended_error_pending);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "skipped_zero".to_string(),
            statement_name: "bad_zero".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert!(!session.portals.contains_key("skipped_zero"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "good".to_string(),
        "SELECT name FROM people WHERE id = $1".to_string(),
        vec![23]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(session.prepared.contains_key("good"));
}

#[test]
fn extended_error_path_binding_rejects_parameter_count_mismatch() {
    let inferred_query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1".to_string(),
        parameter_type_oids: Vec::new(),
    };

    assert_eq!(expected_parameter_count(&inferred_query), 1);
    assert_eq!(
        bind_query_parameters(&inferred_query, &[Some("2".to_string())]),
        Ok("SELECT id FROM people WHERE id = 2".to_string())
    );
    assert_eq!(
        bind_query_parameters(&inferred_query, &[]),
        Err(BindParameterError::CountMismatch)
    );
    assert_eq!(
        bind_query_parameters(
            &inferred_query,
            &[Some("2".to_string()), Some("extra".to_string())]
        ),
        Err(BindParameterError::CountMismatch)
    );

    let typed_query = PreparedQuery {
        query: "SELECT id FROM people".to_string(),
        parameter_type_oids: vec![23],
    };
    assert_eq!(expected_parameter_count(&typed_query), 1);
    assert_eq!(
        bind_query_parameters(&typed_query, &[]),
        Err(BindParameterError::CountMismatch)
    );
}

#[test]
fn extended_parse_infers_supported_parameter_types_from_select_shape() {
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

    assert_eq!(
        resolve_prepared_parameter_type_oids(
            &session,
            "SELECT id FROM people WHERE id > $1 AND name = $2 LIMIT $3",
            Vec::new(),
        ),
        vec![23, 25, 23]
    );
    assert_eq!(
        resolve_prepared_parameter_type_oids(
            &session,
            "SELECT id FROM people WHERE $1 <= id AND $2 = name LIMIT $3",
            Vec::new(),
        ),
        vec![23, 25, 23]
    );
    assert_eq!(
        resolve_prepared_parameter_type_oids(
            &session,
            "SELECT id FROM people WHERE id = $1",
            vec![25],
        ),
        vec![25]
    );
}

#[test]
fn extended_bind_rejects_invalid_values_for_inferred_int4_parameters() {
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };

    assert_eq!(
        bind_query_parameters(&query, &[Some("not-an-int".to_string())]),
        Err(BindParameterError::InvalidTextRepresentation {
            oid: 23,
            value: "not-an-int".to_string(),
        })
    );
}

#[test]
fn extended_bind_rejects_invalid_int4_values_without_installing_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_int4_portal".to_string(),
        "lookup".to_string(),
        vec![0],
        vec![Some(b"not-an-int".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("bad_int4_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "good_int4_portal".to_string(),
        "lookup".to_string(),
        vec![0],
        vec![Some(b"1".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("good_int4_portal"));
}

#[test]
fn extended_execute_supports_parameterized_cursor_declarations() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "".to_string(),
        "DECLARE _psql_cursor NO SCROLL CURSOR FOR SELECT id, name FROM people WHERE id > $1 ORDER BY id".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "".to_string(),
        "".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_execute(&mut writer, &mut session, "", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    let cursor = session.cursors.get("_psql_cursor").unwrap();
    assert_eq!(cursor.rows.len(), 2);
    assert_eq!(
        cursor.rows[0],
        vec![Some("2".to_string()), Some("Linus".to_string())]
    );
}

#[test]
fn extended_cursor_declare_portal_describe_and_close_keep_session_cursor() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "cursor_stmt".to_string(),
        "DECLARE raw_cursor CURSOR FOR SELECT id, name FROM people WHERE id > $1 ORDER BY id"
            .to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "cursor_portal".to_string(),
        "cursor_stmt".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "cursor_portal",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'n']);

    assert!(!handle_execute(&mut writer, &mut session, "cursor_portal", 1).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(session.cursors.contains_key("raw_cursor"));

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "cursor_portal",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(!session.portals.contains_key("cursor_portal"));
    assert!(session.cursors.contains_key("raw_cursor"));

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Statement,
        "cursor_stmt",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(!session.prepared.contains_key("cursor_stmt"));
    assert!(session.cursors.contains_key("raw_cursor"));

    execute_statement(
        &mut writer,
        &mut session,
        "FETCH FORWARD 1 FROM raw_cursor",
        true,
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 3);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C']
    );
    assert_eq!(messages[2].1, b"FETCH 1\0".to_vec());
    assert_eq!(session.cursors.get("raw_cursor").unwrap().position, 1);
}

#[test]
fn extended_cursor_declaration_supports_sql_prepared_execute_binds() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT id, name FROM people WHERE id >= $1 ORDER BY id".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "cursor_exec_stmt".to_string(),
        "DECLARE raw_exec_cursor CURSOR FOR EXECUTE lookup($1)".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    let query = match session.prepared.get("cursor_exec_stmt") {
        Some(PreparedStatement::Extended(query)) => query,
        _ => panic!("expected extended cursor declaration statement"),
    };
    assert_eq!(
        query.parameter_type_oids,
        vec![SqlType::Int4.postgres_oid()]
    );

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "cursor_exec_portal".to_string(),
        "cursor_exec_stmt".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "cursor_exec_portal",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'n']);

    assert!(!handle_execute(&mut writer, &mut session, "cursor_exec_portal", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    let cursor = session.cursors.get("raw_exec_cursor").unwrap();
    assert_eq!(cursor.rows.len(), 2);
    assert_eq!(
        cursor.rows[0],
        vec![Some("2".to_string()), Some("Linus".to_string())]
    );
}

#[test]
fn extended_cursor_sql_execute_parse_rejects_conflicting_explicit_oid() {
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
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT id, name FROM people WHERE id >= $1 ORDER BY id".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "cursor_exec_stmt".to_string(),
            query: "DECLARE raw_exec_cursor CURSOR FOR EXECUTE lookup($1)".to_string(),
            parameter_type_oids: vec![SqlType::Text.postgres_oid()],
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("42P08".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("inconsistent parameter types for SQL EXECUTE placeholder".to_string())
    );
    assert!(extended_error_pending);
    assert!(!session.prepared.contains_key("cursor_exec_stmt"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "skipped_cursor_portal".to_string(),
            statement_name: "cursor_exec_stmt".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert!(!session.portals.contains_key("skipped_cursor_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "cursor_exec_stmt".to_string(),
        "DECLARE raw_exec_cursor CURSOR FOR EXECUTE lookup($1)".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    let query = match session.prepared.get("cursor_exec_stmt") {
        Some(PreparedStatement::Extended(query)) => query,
        _ => panic!("expected recovered extended cursor declaration statement"),
    };
    assert_eq!(
        query.parameter_type_oids,
        vec![SqlType::Int4.postgres_oid()]
    );
}

#[test]
fn extended_cursor_declaration_rejects_malformed_result_format_arity() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "cursor_stmt".to_string(),
        "DECLARE raw_cursor CURSOR FOR SELECT id, name FROM people ORDER BY id".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_cursor_portal".to_string(),
        "cursor_stmt".to_string(),
        Vec::new(),
        Vec::new(),
        vec![0, 0],
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("08P01".to_string())
    );
    assert!(!session.portals.contains_key("bad_cursor_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "good_cursor_portal".to_string(),
        "cursor_stmt".to_string(),
        Vec::new(),
        Vec::new(),
        vec![0],
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_execute(&mut writer, &mut session, "good_cursor_portal", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(session.cursors.contains_key("raw_cursor"));
}

#[test]
fn extended_prepared_portal_lifecycle_closes_session_local_state() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Extended(query.clone()),
    );
    session.portals.insert(
        "lookup_portal".to_string(),
        Portal {
            statement_name: "lookup".to_string(),
            query: query.clone(),
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    session.portals.insert(
        "other_portal".to_string(),
        Portal {
            statement_name: "other".to_string(),
            query: query.clone(),
            parameters: vec![Some("2".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );

    session.close_extended_target(DescribeTarget::Portal, "other_portal");
    assert!(session.prepared.contains_key("lookup"));
    assert!(session.portals.contains_key("lookup_portal"));
    assert!(!session.portals.contains_key("other_portal"));

    session.close_extended_target(DescribeTarget::Statement, "lookup");
    assert!(!session.prepared.contains_key("lookup"));
    assert!(!session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_close_accepts_missing_statement_and_portal_names() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query.clone());
    session.replace_extended_portal(
        "lookup_portal".to_string(),
        Portal {
            statement_name: "lookup".to_string(),
            query,
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Statement,
        "missing_statement"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(session.prepared.contains_key("lookup"));
    assert!(session.portals.contains_key("lookup_portal"));

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "missing_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(session.prepared.contains_key("lookup"));
    assert!(session.portals.contains_key("lookup_portal"));

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "lookup_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(session.prepared.contains_key("lookup"));
    assert!(!session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_close_statement_cascades_to_portals_before_execute_recovery() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query.clone());
    session.replace_extended_portal(
        "lookup_portal".to_string(),
        Portal {
            statement_name: "lookup".to_string(),
            query,
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Statement,
        "lookup"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(!session.prepared.contains_key("lookup"));
    assert!(!session.portals.contains_key("lookup_portal"));

    assert!(handle_execute(&mut writer, &mut session, "lookup_portal", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "lookup_again".to_string(),
        "SELECT name FROM people WHERE id = $1".to_string(),
        vec![23]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "lookup_again_portal".to_string(),
        "lookup_again".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!handle_execute(&mut writer, &mut session, "lookup_again_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_close_portal_removes_only_portal_before_execute_recovery() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query.clone());
    session.replace_extended_portal(
        "closed_portal".to_string(),
        Portal {
            statement_name: "lookup".to_string(),
            query,
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Close {
            target: DescribeTarget::Portal,
            name: "closed_portal".to_string(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(!extended_error_pending);
    assert!(session.prepared.contains_key("lookup"));
    assert!(!session.portals.contains_key("closed_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "closed_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("34000".to_string())
    );
    assert!(extended_error_pending);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_closed_portal (id INT)".to_string()),
    )
    .unwrap();
    assert!(!session.tables.contains_key("skipped_closed_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "rebound_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"2".to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "rebound_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_describe_missing_targets_skip_until_sync_and_recover() {
    for (target, missing_name, expected_code, suffix) in [
        (
            DescribeTarget::Statement,
            "missing_statement",
            "26000",
            "statement",
        ),
        (DescribeTarget::Portal, "missing_portal", "34000", "portal"),
    ] {
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
                rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
                check_constraints: Vec::new(),
                foreign_keys: Vec::new(),
            },
        );
        session.replace_extended_statement(
            "lookup".to_string(),
            PreparedQuery {
                query: "SELECT name FROM people WHERE id = $1".to_string(),
                parameter_type_oids: vec![23],
            },
        );
        let (mut writer, mut reader) = tcp_pair();
        let mut extended_error_pending = false;

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Describe {
                target,
                name: missing_name.to_string(),
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 1);
        assert_eq!(messages[0].0, b'E');
        assert_eq!(
            error_field_value(&messages[0].1, b'C'),
            Some(expected_code.to_string())
        );
        assert!(extended_error_pending);

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::SimpleQuery(format!(
                "CREATE TABLE skipped_describe_{suffix} (id INT)"
            )),
        )
        .unwrap();
        assert!(!session
            .tables
            .contains_key(&format!("skipped_describe_{suffix}")));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Sync,
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
        assert!(!extended_error_pending);

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("recovered_{suffix}_portal"),
                statement_name: "lookup".to_string(),
                parameter_format_codes: Vec::new(),
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Execute {
                portal_name: format!("recovered_{suffix}_portal"),
                max_rows: 0,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 2);
        assert_eq!(
            messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'D', b'C']
        );
        assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
    }
}

#[test]
fn extended_close_missing_targets_do_not_enter_error_recovery() {
    for (target, missing_name, suffix) in [
        (DescribeTarget::Statement, "missing_statement", "statement"),
        (DescribeTarget::Portal, "missing_portal", "portal"),
    ] {
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
                rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
                check_constraints: Vec::new(),
                foreign_keys: Vec::new(),
            },
        );
        session.replace_extended_statement(
            "lookup".to_string(),
            PreparedQuery {
                query: "SELECT name FROM people WHERE id = $1".to_string(),
                parameter_type_oids: vec![23],
            },
        );
        let (mut writer, mut reader) = tcp_pair();
        let mut extended_error_pending = false;

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Close {
                target,
                name: missing_name.to_string(),
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 1);
        assert_eq!(messages[0].0, b'3');
        assert!(!extended_error_pending);
        assert!(session.prepared.contains_key("lookup"));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::SimpleQuery(format!("CREATE TABLE skipped_close_{suffix} (id INT)")),
        )
        .unwrap();
        assert!(session
            .tables
            .contains_key(&format!("skipped_close_{suffix}")));
        assert_eq!(
            read_backend_tags(&mut reader, 2),
            vec![b'C', b'Z'],
            "simple query after missing Close should not be skipped"
        );

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("recovered_{suffix}_portal"),
                statement_name: "lookup".to_string(),
                parameter_format_codes: Vec::new(),
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Execute {
                portal_name: format!("recovered_{suffix}_portal"),
                max_rows: 0,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 2);
        assert_eq!(
            messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'D', b'C']
        );
        assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
    }
}

#[test]
fn extended_execute_missing_portal_skips_until_sync_and_recovers() {
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
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_statement(
        "lookup".to_string(),
        PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        },
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "missing_portal".to_string(),
            max_rows: 1,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("34000".to_string())
    );
    assert!(extended_error_pending);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_execute (id INT)".to_string()),
    )
    .unwrap();
    assert!(!session.tables.contains_key("skipped_execute"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "recovered_execute_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "recovered_execute_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_close_does_not_remove_sql_prepared_statements() {
    let mut session = Session::default();
    session
        .prepared
        .insert("golden_stmt".to_string(), PreparedStatement::AddTen);
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Statement,
        "golden_stmt"
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert_eq!(
        session.prepared.get("golden_stmt"),
        Some(&PreparedStatement::AddTen)
    );
}

#[test]
fn extended_parse_rejects_named_duplicates_and_replaces_unnamed_state() {
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
    let first = PreparedQuery {
        query: "SELECT id FROM people".to_string(),
        parameter_type_oids: Vec::new(),
    };
    session.replace_extended_statement("named".to_string(), first.clone());
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        "named".to_string(),
        "SELECT name FROM people".to_string(),
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert_eq!(
        session.prepared.get("named"),
        Some(&PreparedStatement::Extended(first))
    );

    let unnamed = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement(String::new(), unnamed);
    session.replace_extended_portal(
        String::new(),
        Portal {
            statement_name: String::new(),
            query: PreparedQuery {
                query: "SELECT id FROM people WHERE id = $1".to_string(),
                parameter_type_oids: vec![23],
            },
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "SELECT name FROM people WHERE name = $1".to_string(),
        vec![25]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!session.portals.contains_key(""));
    assert_eq!(
        session.prepared.get(""),
        Some(&PreparedStatement::Extended(PreparedQuery {
            query: "SELECT name FROM people WHERE name = $1".to_string(),
            parameter_type_oids: vec![25],
        }))
    );
}

#[test]
fn extended_parse_accepts_bounded_dml_statements() {
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
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "insert_people".to_string(),
        "INSERT INTO people (id, name) VALUES ($1, $2)".to_string(),
        Vec::new()
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert_eq!(
        session.prepared.get("insert_people"),
        Some(&PreparedStatement::Extended(PreparedQuery {
            query: "INSERT INTO people (id, name) VALUES ($1, $2)".to_string(),
            parameter_type_oids: vec![23, 25],
        }))
    );
}

#[test]
fn extended_dml_insert_update_delete_execute_and_recover() {
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
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "insert_people".to_string(),
        "INSERT INTO people (id, name) VALUES ($1, $2)".to_string(),
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "insert_portal".to_string(),
        "insert_people".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec()), Some(b"Ada".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!handle_execute(&mut writer, &mut session, "insert_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"INSERT 0 1\0");

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "update_people".to_string(),
        "UPDATE people SET name = $1 WHERE id = $2".to_string(),
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "update_portal".to_string(),
        "update_people".to_string(),
        Vec::new(),
        vec![Some(b"Grace".to_vec()), Some(b"1".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!handle_execute(&mut writer, &mut session, "update_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"UPDATE 1\0");

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "delete_people".to_string(),
        "DELETE FROM people WHERE name = $1".to_string(),
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "delete_portal".to_string(),
        "delete_people".to_string(),
        Vec::new(),
        vec![Some(b"Grace".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!handle_execute(&mut writer, &mut session, "delete_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"DELETE 1\0");
    assert!(session.tables["people"].rows.is_empty());
}

#[test]
fn extended_parse_rejects_unsupported_copy_explicitly_without_installing_statement() {
    let mut session = Session::default();
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        "copy_people".to_string(),
        "/* comment */ COPY (SELECT * FROM people) TO STDOUT".to_string(),
        Vec::new()
    )
    .unwrap());

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("0A000".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("COPY is not supported by the compatibility endpoint".to_string())
    );
    assert!(!session.prepared.contains_key("copy_people"));
}

#[test]
fn extended_parse_rejects_unsupported_cursor_options_without_installing_statement() {
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
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
            }],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        "bad_cursor_options".to_string(),
        "DECLARE bad_cursor BINARY CURSOR FOR SELECT id FROM people".to_string(),
        Vec::new()
    )
    .unwrap());

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some(
            "cursor declaration options are not supported by the compatibility endpoint"
                .to_string()
        )
    );
    assert!(!session.prepared.contains_key("bad_cursor_options"));
}

#[test]
fn extended_parse_rejects_extra_parameter_type_oids_without_installing_statement() {
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
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        "lookup".to_string(),
        "SELECT name FROM people WHERE id = $1".to_string(),
        vec![23, 25]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.prepared.contains_key("lookup"));

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "lookup".to_string(),
        "SELECT name FROM people WHERE id = $1".to_string(),
        vec![23]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert_eq!(
        session.prepared.get("lookup"),
        Some(&PreparedStatement::Extended(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        }))
    );
}

#[test]
fn extended_parse_errors_skip_until_sync_and_recover() {
    for (statement_name, query, parameter_type_oids, expected_code, expected_message, suffix) in [
        (
            "bad_type",
            "SELECT name FROM people WHERE id = $1",
            vec![16],
            "0A000",
            "only text and int4 extended-query parameters are supported",
            "bad_type",
        ),
        (
            "too_many_oids",
            "SELECT name FROM people WHERE id = $1",
            vec![23, 25],
            "08P01",
            "parse message has too many parameter type oids",
            "too_many_oids",
        ),
        (
            "unsupported_update_without_where",
            "UPDATE people SET name = $1",
            vec![25],
            "0A000",
            "extended query protocol only supports relational SELECT",
            "unsupported_update_without_where",
        ),
    ] {
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
                rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
                check_constraints: Vec::new(),
                foreign_keys: Vec::new(),
            },
        );
        let (mut writer, mut reader) = tcp_pair();
        let mut extended_error_pending = false;

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Parse {
                statement_name: statement_name.to_string(),
                query: query.to_string(),
                parameter_type_oids,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 1);
        assert_eq!(messages[0].0, b'E');
        assert_eq!(
            error_field_value(&messages[0].1, b'C'),
            Some(expected_code.to_string())
        );
        assert_eq!(
            error_field_value(&messages[0].1, b'M'),
            Some(expected_message.to_string())
        );
        assert!(extended_error_pending);
        assert!(!session.prepared.contains_key(statement_name));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::SimpleQuery(format!("CREATE TABLE skipped_{suffix} (id INT)")),
        )
        .unwrap();
        assert!(!session.tables.contains_key(&format!("skipped_{suffix}")));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Sync,
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
        assert!(!extended_error_pending);

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Parse {
                statement_name: format!("recovered_{suffix}"),
                query: "SELECT name FROM people WHERE id = $1".to_string(),
                parameter_type_oids: vec![23],
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("recovered_{suffix}_portal"),
                statement_name: format!("recovered_{suffix}"),
                parameter_format_codes: Vec::new(),
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Execute {
                portal_name: format!("recovered_{suffix}_portal"),
                max_rows: 0,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 2);
        assert_eq!(
            messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'D', b'C']
        );
        assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
    }
}

#[test]
fn extended_bind_rejects_named_duplicate_portals_and_replaces_unnamed_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query.clone());
    session.replace_extended_portal(
        "named_portal".to_string(),
        Portal {
            statement_name: "lookup".to_string(),
            query: query.clone(),
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "named_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert_eq!(
        session
            .portals
            .get("named_portal")
            .and_then(|portal| portal.parameters.first())
            .cloned(),
        Some(Some("1".to_string()))
    );

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        String::new(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert_eq!(
        session
            .portals
            .get("")
            .and_then(|portal| portal.parameters.first())
            .cloned(),
        Some(Some("2".to_string()))
    );
}

#[test]
fn extended_bind_reports_duplicate_portal_before_payload_errors() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query.clone());
    session.replace_extended_portal(
        "lookup_portal".to_string(),
        Portal {
            statement_name: "lookup".to_string(),
            query,
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "lookup_portal".to_string(),
        "lookup".to_string(),
        vec![1],
        vec![None],
        vec![1]
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("42P03".to_string())
    );
    assert_eq!(
        session
            .portals
            .get("lookup_portal")
            .and_then(|portal| portal.parameters.first())
            .cloned(),
        Some(Some("1".to_string()))
    );

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "fresh_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("fresh_portal"));
}

#[test]
fn extended_bind_rejects_invalid_utf8_text_parameters_without_installing_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE name = $1".to_string(),
        parameter_type_oids: vec![25],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "lookup_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(vec![0xff])],
        Vec::new()
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_bind_rejects_null_parameters_without_installing_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE name = $1".to_string(),
        parameter_type_oids: vec![25],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "lookup_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![None],
        Vec::new()
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_bind_reports_missing_statement_before_format_errors() {
    let mut session = Session::default();
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "missing_portal".to_string(),
        "missing_stmt".to_string(),
        vec![1],
        vec![Some(b"1".to_vec())],
        vec![1]
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("26000".to_string())
    );
    assert!(!session.portals.contains_key("missing_portal"));

    session.replace_extended_statement(
        "lookup".to_string(),
        PreparedQuery {
            query: "SELECT id FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        },
    );
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "lookup_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_bind_accepts_binary_int4_parameters_and_recovers() {
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
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_statement(
        "lookup".to_string(),
        PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        },
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "binary_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(1_i32.to_be_bytes().to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!extended_error_pending);
    assert_eq!(
        session
            .portals
            .get("binary_portal")
            .and_then(|portal| portal.parameters.first())
            .cloned(),
        Some(Some("1".to_string()))
    );

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "binary_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "text_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("text_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "text_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_bind_accepts_binary_int4_and_text_result_formats() {
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
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_statement(
        "lookup".to_string(),
        PreparedQuery {
            query: "SELECT id, name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        },
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "binary_result_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: vec![1, 1],
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!extended_error_pending);
    assert!(session.portals.contains_key("binary_result_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "binary_result_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(
        messages[0].1,
        [
            2_i16.to_be_bytes().as_slice(),
            4_i32.to_be_bytes().as_slice(),
            1_i32.to_be_bytes().as_slice(),
            3_i32.to_be_bytes().as_slice(),
            b"Ada".as_slice(),
        ]
        .concat()
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "text_result_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: vec![0],
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("text_result_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "text_result_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_bind_format_code_count_errors_skip_until_sync_and_recover() {
    for (parameter_format_codes, result_format_codes, expected_message, suffix) in [
        (
            vec![0, 0, 0],
            Vec::new(),
            "bind message has wrong number of parameter format codes",
            "parameter_format_count",
        ),
        (
            Vec::new(),
            vec![0, 0, 0],
            "bind message has wrong number of result format codes",
            "result_format_count",
        ),
    ] {
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
                rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
                check_constraints: Vec::new(),
                foreign_keys: Vec::new(),
            },
        );
        session.replace_extended_statement(
            "lookup".to_string(),
            PreparedQuery {
                query: "SELECT id, name FROM people WHERE id = $1".to_string(),
                parameter_type_oids: vec![23],
            },
        );
        let (mut writer, mut reader) = tcp_pair();
        let mut extended_error_pending = false;

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("bad_{suffix}_portal"),
                statement_name: "lookup".to_string(),
                parameter_format_codes,
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 1);
        assert_eq!(messages[0].0, b'E');
        assert_eq!(
            error_field_value(&messages[0].1, b'C'),
            Some("08P01".to_string())
        );
        assert_eq!(
            error_field_value(&messages[0].1, b'M'),
            Some(expected_message.to_string())
        );
        assert!(extended_error_pending);
        assert!(!session
            .portals
            .contains_key(&format!("bad_{suffix}_portal")));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::SimpleQuery(format!("CREATE TABLE skipped_{suffix} (id INT)")),
        )
        .unwrap();
        assert!(!session.tables.contains_key(&format!("skipped_{suffix}")));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Sync,
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
        assert!(!extended_error_pending);

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("recovered_{suffix}_portal"),
                statement_name: "lookup".to_string(),
                parameter_format_codes: Vec::new(),
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes: vec![0, 0],
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
        assert!(session
            .portals
            .contains_key(&format!("recovered_{suffix}_portal")));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Execute {
                portal_name: format!("recovered_{suffix}_portal"),
                max_rows: 0,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 2);
        assert_eq!(
            messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'D', b'C']
        );
        assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
    }
}

#[test]
fn extended_bind_validates_result_format_code_count_for_select_columns() {
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
    let query = PreparedQuery {
        query: "SELECT id, name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_format_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        vec![0, 0, 0]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("bad_format_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "per_column_format_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        vec![0, 0]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("per_column_format_portal"));
}

#[test]
fn extended_bind_validates_result_format_count_for_sql_execute_columns() {
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
    session.prepared.insert(
        "lookup_sql".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT id, name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    session.replace_extended_statement(
        "lookup_exec".to_string(),
        PreparedQuery {
            query: "EXECUTE lookup_sql($1)".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_sql_execute_format_portal".to_string(),
        "lookup_exec".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        vec![0, 0, 0]
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("08P01".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("bind message has wrong number of result format codes".to_string())
    );
    assert!(!session
        .portals
        .contains_key("bad_sql_execute_format_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "good_sql_execute_format_portal".to_string(),
        "lookup_exec".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        vec![0, 0]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session
        .portals
        .contains_key("good_sql_execute_format_portal"));
}

#[test]
fn extended_bind_validates_parameter_format_code_count() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1 AND name = $2".to_string(),
        parameter_type_oids: vec![23, 25],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_parameter_format_portal".to_string(),
        "lookup".to_string(),
        vec![0, 0, 0],
        vec![Some(b"1".to_vec()), Some(b"alice".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("bad_parameter_format_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "per_parameter_format_portal".to_string(),
        "lookup".to_string(),
        vec![0, 0],
        vec![Some(b"1".to_vec()), Some(b"alice".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("per_parameter_format_portal"));
}

#[test]
fn extended_bind_rejects_parameter_count_mismatch_without_installing_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1 AND name = $2".to_string(),
        parameter_type_oids: vec![23, 25],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_count_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        Vec::new()
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("bad_count_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "good_count_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec()), Some(b"Ada".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("good_count_portal"));
}

#[test]
fn extended_portal_execute_max_rows_suspends_and_resumes_select_portal() {
    let mut session = Session::default();
    session.portals.insert(
        "people_portal".to_string(),
        Portal {
            statement_name: "people_stmt".to_string(),
            query: PreparedQuery {
                query: "SELECT id FROM people ORDER BY id".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: Some(SelectResult {
                columns: vec![int4_column("id")],
                rows: vec![
                    vec![Some("1".to_string())],
                    vec![Some("2".to_string())],
                    vec![Some("3".to_string())],
                ],
            }),
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_portal_batch(&mut writer, &mut session, "people_portal", 2).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'D', b'D', b's']);
    let portal = session.portals.get("people_portal").unwrap();
    assert!(!portal.described);
    assert_eq!(portal.position, 2);

    execute_portal_batch(&mut writer, &mut session, "people_portal", 0).unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 3\0".to_vec());
    let portal = session.portals.get("people_portal").unwrap();
    assert_eq!(portal.position, 3);
    assert!(portal.completed);

    execute_portal_batch(&mut writer, &mut session, "people_portal", 1).unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"SELECT 0\0".to_vec());
}

#[test]
fn extended_describe_portal_after_max_rows_suspend_preserves_position() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_portal(
        "people_portal".to_string(),
        Portal {
            statement_name: "people_stmt".to_string(),
            query: PreparedQuery {
                query: "SELECT id, name FROM people ORDER BY id".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 2).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'D', b'D', b's']);
    let portal = session.portals.get("people_portal").unwrap();
    assert!(!portal.described);
    assert_eq!(portal.position, 2);

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "people_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'T']);
    let portal = session.portals.get("people_portal").unwrap();
    assert!(portal.described);
    assert_eq!(portal.position, 2);

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 3\0".to_vec());
    let portal = session.portals.get("people_portal").unwrap();
    assert_eq!(portal.position, 3);
    assert!(portal.completed);
}

#[test]
fn extended_describe_nodata_portal_marks_portal_described() {
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
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
            }],
            rows: vec![vec![SqlValue::Int4(1)]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_portal(
        "cursor_declare_portal".to_string(),
        Portal {
            statement_name: "cursor_declare_stmt".to_string(),
            query: PreparedQuery {
                query: "DECLARE described_cursor CURSOR FOR SELECT id FROM people".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "cursor_declare_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'n']);
    let portal = session.portals.get("cursor_declare_portal").unwrap();
    assert!(portal.described);
    assert!(portal.result.is_none());
    assert_eq!(portal.position, 0);

    assert!(!handle_execute(&mut writer, &mut session, "cursor_declare_portal", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(session.cursors.contains_key("described_cursor"));
}

#[test]
fn extended_execute_zero_max_rows_exhausts_portal_state() {
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
            rows: vec![
                vec![SqlValue::Int4(1)],
                vec![SqlValue::Int4(2)],
                vec![SqlValue::Int4(3)],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.portals.insert(
        "people_portal".to_string(),
        Portal {
            statement_name: "people_stmt".to_string(),
            query: PreparedQuery {
                query: "SELECT id FROM people ORDER BY id".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 4);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'D', b'D', b'C']
    );
    assert_eq!(messages[3].1, b"SELECT 3\0".to_vec());
    let portal = session.portals.get("people_portal").unwrap();
    assert!(!portal.described);
    assert!(portal.result.is_some());
    assert_eq!(portal.position, 3);

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 1).unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"SELECT 0\0".to_vec());
    assert_eq!(session.portals.get("people_portal").unwrap().position, 3);
}

#[test]
fn extended_execute_leaves_row_description_to_explicit_describe() {
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
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
            }],
            rows: vec![vec![SqlValue::Int4(1)]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_portal(
        "people_portal".to_string(),
        Portal {
            statement_name: "people_stmt".to_string(),
            query: PreparedQuery {
                query: "SELECT id FROM people".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "people_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'T']);
    assert!(session.portals.get("people_portal").unwrap().described);

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_describe_uses_catalog_columns_for_bound_selects() {
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

    assert_eq!(
        describe_query_columns(
            &session,
            "SELECT name, id FROM people WHERE id = 2 ORDER BY name LIMIT 1"
        )
        .unwrap(),
        vec![text_column("name"), int4_column("id")]
    );
}

#[test]
fn extended_describe_sql_execute_uses_prepared_select_columns() {
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
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "EXECUTE lookup(1)".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);

    assert!(!handle_describe(&mut writer, &mut session, DescribeTarget::Statement, "").unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b't', b'T']);

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "exec_portal".to_string(),
        String::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_execute(&mut writer, &mut session, "exec_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_sql_execute_bind_infers_sql_prepared_parameter_types() {
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
            rows: vec![vec![SqlValue::Int4(2), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1 AND name = $2".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "EXECUTE lookup($1, $2)".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    let query = match session.prepared.get("") {
        Some(PreparedStatement::Extended(query)) => query,
        _ => panic!("expected unnamed extended prepared statement"),
    };
    assert_eq!(
        query.parameter_type_oids,
        vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()]
    );

    assert!(!handle_describe(&mut writer, &mut session, DescribeTarget::Statement, "").unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b't', b'T']);

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "exec_portal".to_string(),
        String::new(),
        Vec::new(),
        vec![Some(b"2".to_vec()), Some(b"Ada".to_vec())],
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_execute(&mut writer, &mut session, "exec_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_sql_execute_bind_revalidates_sql_prepared_inferred_types() {
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
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT id, name FROM people WHERE name = $1 LIMIT $2".to_string(),
            parameter_type_oids: vec![SqlType::Text.postgres_oid(), SqlType::Int4.postgres_oid()],
        }),
    );
    session.replace_extended_statement(
        "lookup_exec".to_string(),
        PreparedQuery {
            query: "EXECUTE lookup($1, $2)".to_string(),
            parameter_type_oids: vec![0, 0],
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_sql_execute_bind_portal".to_string(),
        "lookup_exec".to_string(),
        Vec::new(),
        vec![Some(b"Ada".to_vec()), Some(b"not-an-int".to_vec())],
        Vec::new(),
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("22P02".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("invalid input syntax for parameter type oid 23: \"not-an-int\"".to_string())
    );
    assert!(!session.portals.contains_key("bad_sql_execute_bind_portal"));
}

#[test]
fn extended_sql_execute_describe_reports_literal_parameter_errors() {
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
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "EXECUTE lookup('not-an-int')".to_string(),
        Vec::new(),
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert!(String::from_utf8_lossy(&messages[0].1)
        .contains("invalid input syntax for parameter type oid 23: \"not-an-int\""));
    assert!(!session.prepared.contains_key(""));
}

#[test]
fn extended_cursor_sql_execute_describe_reports_literal_parameter_errors() {
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
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "DECLARE lookup_cursor CURSOR FOR EXECUTE lookup('not-an-int')".to_string(),
        Vec::new(),
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert!(String::from_utf8_lossy(&messages[0].1)
        .contains("invalid input syntax for parameter type oid 23: \"not-an-int\""));
    assert!(!session.prepared.contains_key(""));
    assert!(!session.cursors.contains_key("lookup_cursor"));
}

#[test]
fn extended_sql_execute_rejects_conflicting_reused_placeholder_types() {
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
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1 AND name = $2".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "EXECUTE lookup($1, $1)".to_string(),
        Vec::new(),
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert!(String::from_utf8_lossy(&messages[0].1)
        .contains("inconsistent parameter types for SQL EXECUTE placeholder"));
    assert!(!session.prepared.contains_key(""));
}

#[test]
fn extended_sql_execute_sparse_outer_placeholders_skip_unused_binds() {
    let query = PreparedQuery {
        query: "EXECUTE lookup($2, 'Ada')".to_string(),
        parameter_type_oids: vec![0, SqlType::Int4.postgres_oid()],
    };

    let bound = bind_query_parameters(
        &query,
        &[
            Some("not-an-int-and-not-used".to_string()),
            Some("2".to_string()),
        ],
    )
    .unwrap();
    assert_eq!(bound, "EXECUTE lookup(2, 'Ada')");

    assert_eq!(
        bind_query_parameters(
            &query,
            &[
                Some("still-not-used".to_string()),
                Some("not-an-int".to_string()),
            ],
        ),
        Err(BindParameterError::InvalidTextRepresentation {
            oid: SqlType::Int4.postgres_oid(),
            value: "not-an-int".to_string(),
        })
    );
}

#[test]
fn sql_prepare_helpers_parse_supported_relational_select_shape() {
    assert_eq!(
        parse_sql_prepare(
            "PREPARE lookup(int4, pg_catalog.text) AS SELECT id, name FROM people WHERE id = $1 AND name = $2",
        ),
        Some((
            "lookup".to_string(),
            vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()],
            "SELECT id, name FROM people WHERE id = $1 AND name = $2".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_prepare(
            r#"PREPARE "lookup(one)"(int4) AS SELECT name FROM people WHERE id = $1"#,
        ),
        Some((
            "lookup(one)".to_string(),
            vec![SqlType::Int4.postgres_oid()],
            "SELECT name FROM people WHERE id = $1".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_prepare(
            r#"PREPARE "lookup as stmt"(int4) AS SELECT name FROM people WHERE id = $1"#,
        ),
        Some((
            "lookup as stmt".to_string(),
            vec![SqlType::Int4.postgres_oid()],
            "SELECT name FROM people WHERE id = $1".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_prepare(
            r#"PREPARE "lookup ""quoted"""(int4) AS SELECT name FROM people WHERE id = $1"#,
        ),
        Some((
            r#"lookup "quoted""#.to_string(),
            vec![SqlType::Int4.postgres_oid()],
            "SELECT name FROM people WHERE id = $1".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_prepare(
            "PREPARE comment_lookup(/* id */ int4, /* name */ text) /* target */ AS SELECT id, name FROM people WHERE id = $1 AND name = $2",
        ),
        Some((
            "comment_lookup".to_string(),
            vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()],
            "SELECT id, name FROM people WHERE id = $1 AND name = $2".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_prepare(
            "PREPARE comment_as_lookup -- AS inside line comment\n\
             (int4, text) /* nested AS /* inner AS */ target */ AS SELECT id, name FROM people WHERE id = $1 AND name = $2",
        ),
        Some((
            "comment_as_lookup".to_string(),
            vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()],
            "SELECT id, name FROM people WHERE id = $1 AND name = $2".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_prepare(
            "PREPARE\n\
             whitespace_lookup\t(int4,\n\
             text)\n\
             AS SELECT id, name FROM people WHERE id = $1 AND name = $2",
        ),
        Some((
            "whitespace_lookup".to_string(),
            vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()],
            "SELECT id, name FROM people WHERE id = $1 AND name = $2".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_prepare(
            "/* leading */ -- prepare follows\n\
             PREPARE leading_comment_lookup(int4) AS SELECT name FROM people WHERE id = $1",
        ),
        Some((
            "leading_comment_lookup".to_string(),
            vec![SqlType::Int4.postgres_oid()],
            "SELECT name FROM people WHERE id = $1".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_prepare_name(
            "PREPARE lookup(jsonb) AS INSERT INTO people (id, name) VALUES ($1, 'Ada')"
        ),
        Some("lookup".to_string())
    );
    assert_eq!(
        parse_sql_prepare_name(
            r#"PREPARE "Mixed Lookup"(jsonb) AS INSERT INTO people (id, name) VALUES ($1, 'Ada')"#
        ),
        Some("Mixed Lookup".to_string())
    );
    assert_eq!(
        parse_sql_prepare_name(
            r#"PREPARE "lookup(one)"(jsonb) AS INSERT INTO people (id, name) VALUES ($1, 'Ada')"#
        ),
        Some("lookup(one)".to_string())
    );
    assert_eq!(
        parse_sql_prepare_name(
            r#"PREPARE "lookup as stmt"(jsonb) AS INSERT INTO people (id, name) VALUES ($1, 'Ada')"#
        ),
        Some("lookup as stmt".to_string())
    );
    assert_eq!(
        parse_sql_prepare_name(
            r#"PREPARE "lookup ""quoted"""(jsonb) AS INSERT INTO people (id, name) VALUES ($1, 'Ada')"#
        ),
        Some(r#"lookup "quoted""#.to_string())
    );
    assert_eq!(
        parse_sql_prepare_name(
            "PREPARE comment_as_lookup -- AS inside line comment\n\
             (jsonb) /* block AS */ AS INSERT INTO people (id, name) VALUES ($1, 'Ada')"
        ),
        Some("comment_as_lookup".to_string())
    );
    assert_eq!(
        parse_sql_prepare_name(
            "/* duplicate probe */ PREPARE leading_comment_lookup(jsonb) AS INSERT INTO people (id, name) VALUES ($1, 'Ada')"
        ),
        Some("leading_comment_lookup".to_string())
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(2, 'O''Brien')"),
        Some((
            "lookup".to_string(),
            vec![Some("2".to_string()), Some("O'Brien".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute(r"EXECUTE lookup(E'O\'Brien', e'line\nfeed')"),
        Some((
            "lookup".to_string(),
            vec![Some("O'Brien".to_string()), Some("line\nfeed".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute(r"EXECUTE lookup(E'Ada\x20Lovelace', E'Grace\040Hopper')"),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada Lovelace".to_string()),
                Some("Grace Hopper".to_string()),
            ],
        ))
    );
    assert_eq!(
        parse_sql_execute(r"EXECUTE lookup(U&'Ada\0020Lovelace', u&'Grace\+000020Hopper')"),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada Lovelace".to_string()),
                Some("Grace Hopper".to_string()),
            ],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(N'Ada''s notes', text n'Grace Hopper')"),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada's notes".to_string()),
                Some("Grace Hopper".to_string()),
            ],
        ))
    );
    assert_eq!(
        parse_sql_execute(
            r"EXECUTE lookup(U&'Ada!0020Lovelace' UESCAPE '!', text U&'Grace~+000020Hopper' UESCAPE '~')"
        ),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada Lovelace".to_string()),
                Some("Grace Hopper".to_string()),
            ],
        ))
    );
    assert_eq!(
        parse_sql_execute(r"EXECUTE lookup($tag$Ada, Lovelace$tag$, $$Grace (Hopper)$$)"),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada, Lovelace".to_string()),
                Some("Grace (Hopper)".to_string()),
            ],
        ))
    );
    assert_eq!(
        parse_sql_execute(
            "EXECUTE lookup($tag$Ada -- not comment$tag$, $$Grace /* not comment */$$)"
        ),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada -- not comment".to_string()),
                Some("Grace /* not comment */".to_string()),
            ],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup((2), ('Linus'))"),
        Some((
            "lookup".to_string(),
            vec![Some("2".to_string()), Some("Linus".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(((2)), (('Linus')))"),
        Some((
            "lookup".to_string(),
            vec![Some("2".to_string()), Some("Linus".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(2::int4, 'Ada'::text)"),
        Some((
            "lookup".to_string(),
            vec![Some("2".to_string()), Some("Ada".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(int4 '2', text 'Ada')"),
        Some((
            "lookup".to_string(),
            vec![Some("2".to_string()), Some("Ada".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(text U&'Ada\\0020Lovelace')"),
        Some(("lookup".to_string(), vec![Some("Ada Lovelace".to_string())],))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup('Ada'\n' Lovelace', text 'Grace'\n' Hopper')"),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada Lovelace".to_string()),
                Some("Grace Hopper".to_string()),
            ],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup('Ada' /* newline\ncomment */ ' Lovelace')"),
        Some(("lookup".to_string(), vec![Some("Ada Lovelace".to_string())],))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup('Grace' -- newline comment\n' Hopper')"),
        Some(("lookup".to_string(), vec![Some("Grace Hopper".to_string())],))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(pg_catalog.int4 '3', pg_catalog.text $$Grace$$)"),
        Some((
            "lookup".to_string(),
            vec![Some("3".to_string()), Some("Grace".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup($1::int4, $2::text)"),
        Some((
            "lookup".to_string(),
            vec![Some("$1".to_string()), Some("$2".to_string())],
        ))
    );
    assert!(parse_sql_execute("EXECUTE lookup(int4 $1, text $2)").is_none());
    assert_eq!(
        parse_sql_execute("EXECUTE lookup((3::pg_catalog.int4), ('Grace')::pg_catalog.text)"),
        Some((
            "lookup".to_string(),
            vec![Some("3".to_string()), Some("Grace".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(CAST($1 AS int4), CAST('Grace' AS text))"),
        Some((
            "lookup".to_string(),
            vec![Some("$1".to_string()), Some("Grace".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute(
            "EXECUTE lookup(CAST(('3') AS pg_catalog.int4), CAST($1 AS pg_catalog.text))"
        ),
        Some((
            "lookup".to_string(),
            vec![Some("3".to_string()), Some("$1".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(CAST($tag$Ada AS text$tag$ AS text))"),
        Some(("lookup".to_string(), vec![Some("Ada AS text".to_string())],))
    );
    assert_eq!(
        parse_sql_execute(
            "EXECUTE lookup($tag$Ada::literal$tag$::text, $$Grace::literal$$::pg_catalog.text)"
        ),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada::literal".to_string()),
                Some("Grace::literal".to_string()),
            ],
        ))
    );
    assert_eq!(
        parse_sql_execute(
            "EXECUTE lookup(($tag$Ada (literal$tag$), CAST(($$Grace ) literal$$) AS text))"
        ),
        Some((
            "lookup".to_string(),
            vec![
                Some("Ada (literal".to_string()),
                Some("Grace ) literal".to_string()),
            ],
        ))
    );
    assert!(parse_sql_execute("EXECUTE lookup(CAST($1 AS jsonb))").is_none());
    assert_eq!(
        parse_sql_execute("EXECUTE comment_lookup(/* id */ 2, /* name */ 'Ada' /* keep */)"),
        Some((
            "comment_lookup".to_string(),
            vec![Some("2".to_string()), Some("Ada".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE\nwhitespace_lookup\t(2,\n'Ada')"),
        Some((
            "whitespace_lookup".to_string(),
            vec![Some("2".to_string()), Some("Ada".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute("-- run prepared\nEXECUTE leading_comment_lookup(/* id */ 2)"),
        Some((
            "leading_comment_lookup".to_string(),
            vec![Some("2".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_execute(r#"EXECUTE "lookup(one)"(2)"#),
        Some(("lookup(one)".to_string(), vec![Some("2".to_string())],))
    );
    assert_eq!(
        parse_sql_execute(r#"EXECUTE "lookup ""quoted"""(2)"#),
        Some((
            r#"lookup "quoted""#.to_string(),
            vec![Some("2".to_string())],
        ))
    );
    assert_eq!(
        parse_sql_prepare("PREPARE lookup_all AS SELECT id, name FROM people ORDER BY id",),
        Some((
            "lookup_all".to_string(),
            Vec::new(),
            "SELECT id, name FROM people ORDER BY id".to_string(),
        ))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup_all"),
        Some(("lookup_all".to_string(), Vec::new()))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup_all()"),
        Some(("lookup_all".to_string(), Vec::new()))
    );
    assert_eq!(
        parse_sql_execute("EXECUTE lookup(NULL, 'Ada')"),
        Some(("lookup".to_string(), vec![None, Some("Ada".to_string())],))
    );
    assert!(matches!(
        parse_sql_deallocate("DEALLOCATE ALL"),
        Some(SqlDeallocateTarget::All)
    ));
    assert!(matches!(
        parse_sql_deallocate("DEALLOCATE PREPARE ALL"),
        Some(SqlDeallocateTarget::All)
    ));
    assert!(matches!(
        parse_sql_deallocate("DEALLOCATE PREPARE lookup"),
        Some(SqlDeallocateTarget::Named(name)) if name == "lookup"
    ));
    assert!(matches!(
        parse_sql_deallocate(r#"DEALLOCATE PREPARE "Mixed Lookup""#),
        Some(SqlDeallocateTarget::Named(name)) if name == "Mixed Lookup"
    ));
    assert!(matches!(
        parse_sql_deallocate(r#"DEALLOCATE PREPARE "lookup ""quoted""""#),
        Some(SqlDeallocateTarget::Named(name)) if name == r#"lookup "quoted""#
    ));
    assert!(matches!(
        parse_sql_deallocate("DEALLOCATE /* scope */ PREPARE /* target */ comment_lookup"),
        Some(SqlDeallocateTarget::Named(name)) if name == "comment_lookup"
    ));
    assert!(matches!(
        parse_sql_deallocate("DEALLOCATE\nPREPARE\twhitespace_lookup"),
        Some(SqlDeallocateTarget::Named(name)) if name == "whitespace_lookup"
    ));
    assert!(matches!(
        parse_sql_deallocate("/* free */ DEALLOCATE PREPARE leading_comment_lookup"),
        Some(SqlDeallocateTarget::Named(name)) if name == "leading_comment_lookup"
    ));
    assert_eq!(
        strip_leading_sql_comments("/* outer /* inner */ done */ -- trailing\nEXECUTE lookup")
            .unwrap(),
        "EXECUTE lookup"
    );
    assert!(parse_sql_execute("/* unterminated EXECUTE lookup").is_none());
    assert!(parse_sql_prepare("PREPARE bad(jsonb) AS SELECT id FROM people").is_none());
    assert!(parse_sql_execute("EXECUTE lookup('unterminated)").is_none());
    assert!(parse_sql_execute("EXECUTE lookup(E'unterminated)").is_none());
    assert!(parse_sql_execute(r"EXECUTE lookup(E'bad\xzz')").is_none());
    assert!(parse_sql_execute("EXECUTE lookup('Ada' 'Lovelace')").is_none());
    assert!(parse_sql_execute("EXECUTE lookup(U&'unterminated)").is_none());
    assert!(parse_sql_execute("EXECUTE lookup(U&'bad\\00xz')").is_none());
    assert!(parse_sql_execute("EXECUTE lookup(U&'bad!00xz' UESCAPE '!')").is_none());
    assert!(parse_sql_execute("EXECUTE lookup(U&'bad!0020' UESCAPE '+')").is_none());
    assert!(parse_sql_execute("EXECUTE lookup($tag$unterminated)").is_none());
    assert!(parse_sql_deallocate("DEALLOCATE PREPARE").is_none());
}

#[test]
fn extended_cursor_fetch_count_helpers_use_supported_select_results() {
    assert_eq!(
        parse_declare_cursor(
            "DECLARE _psql_cursor NO SCROLL CURSOR FOR\nSELECT id, name FROM people ORDER BY id"
        ),
        Some((
            "_psql_cursor".to_string(),
            "select id, name from people order by id".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor("DECLARE Mixed_Cursor CURSOR FOR SELECT id FROM people"),
        Some((
            "mixed_cursor".to_string(),
            "select id from people".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor(r#"DECLARE "Mixed Cursor" CURSOR FOR SELECT id FROM people"#),
        Some((
            "Mixed Cursor".to_string(),
            "select id from people".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor(r#"DECLARE "quote""cursor" CURSOR FOR SELECT id FROM people"#),
        Some((
            r#"quote"cursor"#.to_string(),
            "select id from people".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor(
            "DECLARE _psql_cursor NO SCROLL CURSOR WITHOUT HOLD FOR SELECT id FROM people"
        ),
        Some((
            "_psql_cursor".to_string(),
            "select id from people".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor("DECLARE _psql_cursor CURSOR WITHOUT HOLD FOR SELECT id FROM people"),
        Some((
            "_psql_cursor".to_string(),
            "select id from people".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor(
            "DECLARE _psql_cursor ASENSITIVE NO SCROLL CURSOR FOR SELECT id FROM people"
        ),
        Some((
            "_psql_cursor".to_string(),
            "select id from people".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor(
            r#"DECLARE /* name */ "Comment Cursor" /* sensitivity */ ASENSITIVE /* direction */ NO SCROLL /* kind */ CURSOR /* lifetime */ WITHOUT HOLD /* query */ FOR SELECT id FROM people"#
        ),
        Some((
            "Comment Cursor".to_string(),
            "select id from people".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor(
            "DECLARE _psql_cursor INSENSITIVE CURSOR WITHOUT HOLD FOR SELECT id FROM people"
        ),
        Some((
            "_psql_cursor".to_string(),
            "select id from people".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor(
            "DECLARE _psql_cursor CURSOR FOR\nSELECT id, name FROM people ORDER BY id"
        ),
        Some((
            "_psql_cursor".to_string(),
            "select id, name from people order by id".to_string()
        ))
    );
    assert_eq!(
        parse_declare_cursor("DECLARE _psql_cursor BINARY CURSOR FOR SELECT id FROM people"),
        None
    );
    assert!(is_unsupported_declare_cursor_statement(
        "DECLARE _psql_cursor BINARY CURSOR FOR SELECT id FROM people"
    ));
    assert_eq!(
        parse_declare_cursor("DECLARE _psql_cursor SCROLL CURSOR FOR SELECT id FROM people"),
        None
    );
    assert!(is_unsupported_declare_cursor_statement(
        "DECLARE _psql_cursor SCROLL CURSOR FOR SELECT id FROM people"
    ));
    assert_eq!(
        parse_declare_cursor(
            "DECLARE _psql_cursor NO SCROLL CURSOR WITH HOLD FOR SELECT id FROM people"
        ),
        None
    );
    assert!(is_unsupported_declare_cursor_statement(
        "DECLARE _psql_cursor NO SCROLL CURSOR WITH HOLD FOR SELECT id FROM people"
    ));
    assert_eq!(
        parse_fetch_forward("FETCH FORWARD 2 FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH 0 FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(0)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH 2 FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH FORWARD 2 FROM Mixed_Cursor"),
        Some(("mixed_cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_fetch_forward(
            r#"FETCH /* direction */ FORWARD 2 /* marker */ FROM /* target */ "Mixed Cursor""#
        ),
        Some(("Mixed Cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_fetch_forward(r#"FETCH FORWARD 2 FROM "Mixed Cursor""#),
        Some(("Mixed Cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_fetch_forward(r#"FETCH ALL "Mixed Cursor""#),
        Some(("Mixed Cursor".to_string(), None))
    );
    assert_eq!(
        parse_fetch_forward("FETCH NEXT FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH FORWARD FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH IN _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH FORWARD ALL FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), None))
    );
    assert_eq!(
        parse_fetch_forward("FETCH ALL FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), None))
    );
    assert_eq!(
        parse_fetch_forward("FETCH 1 IN _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH FORWARD _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH FORWARD 2 _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_fetch_forward("FETCH ALL _psql_cursor"),
        Some(("_psql_cursor".to_string(), None))
    );
    assert_eq!(
        parse_move_forward("MOVE FORWARD 2 FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_move_forward("MOVE 0 FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(0)))
    );
    assert_eq!(
        parse_move_forward("MOVE 2 FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_move_forward("MOVE FORWARD 2 FROM MIXED_CURSOR"),
        Some(("mixed_cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_move_forward(
            r#"MOVE /* direction */ FORWARD 2 /* marker */ FROM /* target */ "Mixed Cursor""#
        ),
        Some(("Mixed Cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_move_forward(r#"MOVE FORWARD 2 FROM "Mixed Cursor""#),
        Some(("Mixed Cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_move_forward(r#"MOVE "Mixed Cursor""#),
        Some(("Mixed Cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_move_forward("MOVE NEXT FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_move_forward("MOVE FROM _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_move_forward("MOVE IN _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_move_forward("MOVE FORWARD ALL IN _psql_cursor"),
        Some(("_psql_cursor".to_string(), None))
    );
    assert_eq!(
        parse_move_forward("MOVE _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_move_forward("MOVE FORWARD _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(1)))
    );
    assert_eq!(
        parse_move_forward("MOVE FORWARD 2 _psql_cursor"),
        Some(("_psql_cursor".to_string(), Some(2)))
    );
    assert_eq!(
        parse_move_forward("MOVE ALL _psql_cursor"),
        Some(("_psql_cursor".to_string(), None))
    );
    assert_eq!(
        parse_move_forward("MOVE BACKWARD 1 FROM _psql_cursor"),
        None
    );
    assert!(is_unsupported_move_cursor_statement(
        "MOVE BACKWARD 1 FROM _psql_cursor"
    ));
    assert!(is_unsupported_move_cursor_statement(
        "MOVE BACKWARD 1 _psql_cursor"
    ));
    assert!(is_unsupported_move_cursor_statement(
        "MOVE FIRST FROM _psql_cursor"
    ));
    assert!(is_unsupported_move_cursor_statement(
        "MOVE LAST _psql_cursor"
    ));
    assert!(is_unsupported_move_cursor_statement(
        "MOVE ABSOLUTE 3 FROM _psql_cursor"
    ));
    assert!(is_unsupported_move_cursor_statement(
        "MOVE RELATIVE 2 _psql_cursor"
    ));
    assert_eq!(
        parse_fetch_forward("FETCH BACKWARD 1 FROM _psql_cursor"),
        None
    );
    assert!(is_unsupported_fetch_cursor_statement(
        "FETCH BACKWARD 1 FROM _psql_cursor"
    ));
    assert!(is_unsupported_fetch_cursor_statement(
        "FETCH BACKWARD 1 _psql_cursor"
    ));
    assert!(is_unsupported_fetch_cursor_statement(
        "FETCH FIRST FROM _psql_cursor"
    ));
    assert!(is_unsupported_fetch_cursor_statement(
        "FETCH LAST _psql_cursor"
    ));
    assert!(is_unsupported_fetch_cursor_statement(
        "FETCH ABSOLUTE 3 FROM _psql_cursor"
    ));
    assert!(is_unsupported_fetch_cursor_statement(
        "FETCH RELATIVE 2 _psql_cursor"
    ));
    assert_eq!(
        parse_close_cursor("CLOSE _psql_cursor"),
        Some(CloseCursorTarget::Named("_psql_cursor".to_string()))
    );
    assert_eq!(
        parse_close_cursor("CLOSE Mixed_Cursor"),
        Some(CloseCursorTarget::Named("mixed_cursor".to_string()))
    );
    assert_eq!(
        parse_close_cursor(r#"CLOSE "Mixed Cursor""#),
        Some(CloseCursorTarget::Named("Mixed Cursor".to_string()))
    );
    assert_eq!(
        parse_close_cursor(r#"CLOSE /* target */ "Mixed Cursor""#),
        Some(CloseCursorTarget::Named("Mixed Cursor".to_string()))
    );
    assert_eq!(parse_close_cursor(r#"CLOSE "Mixed Cursor"#), None);
    assert_eq!(
        parse_close_cursor("CLOSE ALL"),
        Some(CloseCursorTarget::All)
    );
    assert_eq!(
        max_placeholder_index("select id from people where id > $1"),
        1
    );
    assert_eq!(
        replace_parameter_placeholders_with_dummy_literals(
            "select id from people where id > $1 limit $2"
        ),
        "select id from people where id > 1 limit 1"
    );

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
            rows: vec![
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let Command::Select(select) = parse_command("select id, name from people order by id").unwrap()
    else {
        panic!("expected supported SELECT");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(result.columns, vec![int4_column("id"), text_column("name")]);
    assert_eq!(
        result.rows,
        vec![
            vec![Some("1".to_string()), Some("Ada".to_string())],
            vec![Some("2".to_string()), Some("Linus".to_string())],
            vec![Some("3".to_string()), Some("Grace".to_string())],
        ]
    );
    let Command::Select(select) =
        parse_command("select id, name from people order by id limit 1 offset 1").unwrap()
    else {
        panic!("expected supported SELECT");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![Some("2".to_string()), Some("Linus".to_string())]]
    );
    session
        .tables
        .get_mut("people")
        .unwrap()
        .rows
        .push(vec![SqlValue::Int4(4), SqlValue::Text("Grace".to_string())]);
    let Command::Select(select) =
        parse_command("select distinct name from people order by name desc limit 2 offset 1")
            .unwrap()
    else {
        panic!("expected supported SELECT");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(result.columns, vec![text_column("name")]);
    assert_eq!(
        result.rows,
        vec![
            vec![Some("Grace".to_string())],
            vec![Some("Ada".to_string())],
        ]
    );
    let Command::Select(select) =
        parse_command("select distinct name from people order by id").unwrap()
    else {
        panic!("expected supported SELECT parse");
    };
    let err = execute_select_result(&session, &select).unwrap_err();
    assert_eq!(err.code, "0A000");
    assert_eq!(
        err.message,
        "SELECT DISTINCT ORDER BY must reference a selected column"
    );
    let Command::Select(select) =
        parse_command("select name, count(*) from people group by name order by name").unwrap()
    else {
        panic!("expected supported SELECT aggregate parse");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(
        result.columns,
        vec![text_column("name"), int8_column("count")]
    );
    assert_eq!(
        result.rows,
        vec![
            vec![Some("Ada".to_string()), Some("1".to_string())],
            vec![Some("Grace".to_string()), Some("2".to_string())],
            vec![Some("Linus".to_string()), Some("1".to_string())],
        ]
    );
    let Command::Select(select) =
        parse_command("select count(*) from people where id >= 2").unwrap()
    else {
        panic!("expected supported SELECT aggregate parse");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(result.columns, vec![int8_column("count")]);
    assert_eq!(result.rows, vec![vec![Some("3".to_string())]]);
    let Command::Select(select) =
        parse_command("select name, sum(id) from people group by name order by sum desc").unwrap()
    else {
        panic!("expected supported SELECT sum aggregate parse");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(
        result.columns,
        vec![text_column("name"), int8_column("sum")]
    );
    assert_eq!(
        result.rows,
        vec![
            vec![Some("Grace".to_string()), Some("7".to_string())],
            vec![Some("Linus".to_string()), Some("2".to_string())],
            vec![Some("Ada".to_string()), Some("1".to_string())],
        ]
    );
    let Command::Select(select) =
        parse_command("select sum(id) from people where name = 'Grace'").unwrap()
    else {
        panic!("expected supported SELECT sum aggregate parse");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(result.columns, vec![int8_column("sum")]);
    assert_eq!(result.rows, vec![vec![Some("7".to_string())]]);
    let Command::Select(select) = parse_command("select sum(name) from people").unwrap() else {
        panic!("expected supported SELECT sum aggregate parse");
    };
    let err = execute_select_result(&session, &select).unwrap_err();
    assert_eq!(err.code, "0A000");
    assert_eq!(err.message, "SUM only supports int4 columns");
    let Command::Select(select) =
        parse_command("select name, avg(id) from people group by name order by avg desc").unwrap()
    else {
        panic!("expected supported SELECT avg aggregate parse");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(
        result.columns,
        vec![text_column("name"), numeric_column("avg")]
    );
    assert_eq!(
        result.rows,
        vec![
            vec![
                Some("Grace".to_string()),
                Some("3.5000000000000000".to_string())
            ],
            vec![
                Some("Linus".to_string()),
                Some("2.0000000000000000".to_string())
            ],
            vec![
                Some("Ada".to_string()),
                Some("1.0000000000000000".to_string())
            ],
        ]
    );
    let Command::Select(select) =
        parse_command("select avg(id) from people where name = 'Grace'").unwrap()
    else {
        panic!("expected supported SELECT avg aggregate parse");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(result.columns, vec![numeric_column("avg")]);
    assert_eq!(
        result.rows,
        vec![vec![Some("3.5000000000000000".to_string())]]
    );
    let Command::Select(select) = parse_command("select avg(name) from people").unwrap() else {
        panic!("expected supported SELECT avg aggregate parse");
    };
    let err = execute_select_result(&session, &select).unwrap_err();
    assert_eq!(err.code, "0A000");
    assert_eq!(err.message, "AVG only supports int4 columns");
    let Command::Select(select) =
        parse_command("select name, min(id) from people group by name order by min desc").unwrap()
    else {
        panic!("expected supported SELECT min aggregate parse");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(
        result.columns,
        vec![text_column("name"), int4_column("min")]
    );
    assert_eq!(
        result.rows,
        vec![
            vec![Some("Grace".to_string()), Some("3".to_string())],
            vec![Some("Linus".to_string()), Some("2".to_string())],
            vec![Some("Ada".to_string()), Some("1".to_string())],
        ]
    );
    let Command::Select(select) =
        parse_command("select max(name) from people where id <= 2").unwrap()
    else {
        panic!("expected supported SELECT max aggregate parse");
    };
    let result = execute_select_result(&session, &select).unwrap();
    assert_eq!(result.columns, vec![text_column("max")]);
    assert_eq!(result.rows, vec![vec![Some("Linus".to_string())]]);
    let Command::Select(select) =
        parse_command("select name, max(id) from people order by name").unwrap()
    else {
        panic!("expected supported SELECT grouped max aggregate parse");
    };
    let err = execute_select_result(&session, &select).unwrap_err();
    assert_eq!(err.code, "0A000");
    assert_eq!(err.message, "grouped MIN/MAX requires GROUP BY");

    assert_eq!(
        describe_parameterized_select_shape(
            "select id from people where id > $1 order by id limit $2"
        ),
        Some((
            "people".to_string(),
            SelectProjection::Columns(vec!["id".to_string()])
        ))
    );
    assert_eq!(
        describe_parameterized_select_shape(
            "select id from people where id > $1 order by id limit $2 offset $3"
        ),
        Some((
            "people".to_string(),
            SelectProjection::Columns(vec!["id".to_string()])
        ))
    );
    assert_eq!(
        describe_parameterized_select_shape(
            "select people.id from people join pets on people.id = pets.owner_id where people.id = $1"
        ),
        None
    );
}

#[test]
fn declare_cursor_executes_sql_prepared_select_results() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Grace".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT id, name FROM people WHERE id >= $1 ORDER BY id".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!execute_declare_cursor(
        &mut writer,
        &mut session,
        "_psql_cursor".to_string(),
        "EXECUTE lookup(1)",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    let cursor = session.cursors.get("_psql_cursor").unwrap();
    assert_eq!(cursor.columns, vec![int4_column("id"), text_column("name")]);
    assert_eq!(cursor.rows.len(), 2);

    execute_fetch_forward(&mut writer, &mut session, "_psql_cursor", Some(1)).unwrap();
    let messages = read_backend_messages(&mut reader, 3);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C']
    );
    assert_eq!(messages[2].1, b"FETCH 1\0".to_vec());
    assert_eq!(session.cursors.get("_psql_cursor").unwrap().position, 1);
}

#[test]
fn declare_cursor_rejects_duplicate_name_without_replacing_existing_cursor() {
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
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_declare_cursor(
        &mut writer,
        &mut session,
        "dup_cursor".to_string(),
        "select id, name from people order by id",
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert_eq!(session.cursors.get("dup_cursor").unwrap().rows.len(), 2);

    execute_declare_cursor(
        &mut writer,
        &mut session,
        "dup_cursor".to_string(),
        "select id, name from people where id = 2",
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);

    let cursor = session.cursors.get("dup_cursor").unwrap();
    assert_eq!(cursor.rows.len(), 2);
    assert_eq!(cursor.position, 0);
}

#[test]
fn extended_cursor_move_forward_advances_without_rows() {
    let mut session = Session::default();
    session.cursors.insert(
        "live_cursor".to_string(),
        Cursor {
            columns: vec![int4_column("id")],
            rows: vec![
                vec![Some("1".to_string())],
                vec![Some("2".to_string())],
                vec![Some("3".to_string())],
            ],
            position: 0,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_move_forward(&mut writer, &mut session, "live_cursor", Some(2)).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert_eq!(session.cursors.get("live_cursor").unwrap().position, 2);

    execute_fetch_forward(&mut writer, &mut session, "live_cursor", Some(1)).unwrap();
    let messages = read_backend_messages(&mut reader, 3);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C']
    );
    assert_eq!(messages[2].1, b"FETCH 1\0".to_vec());
    assert_eq!(session.cursors.get("live_cursor").unwrap().position, 3);

    execute_move_forward(&mut writer, &mut session, "live_cursor", None).unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].1, b"MOVE 0\0".to_vec());
}

#[test]
fn extended_cursor_fetch_all_consumes_remaining_rows() {
    let mut session = Session::default();
    session.cursors.insert(
        "live_cursor".to_string(),
        Cursor {
            columns: vec![int4_column("id")],
            rows: vec![
                vec![Some("1".to_string())],
                vec![Some("2".to_string())],
                vec![Some("3".to_string())],
            ],
            position: 1,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_fetch_forward(&mut writer, &mut session, "live_cursor", None).unwrap();
    let messages = read_backend_messages(&mut reader, 4);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'D', b'C']
    );
    assert_eq!(messages[3].1, b"FETCH 2\0".to_vec());
    assert_eq!(session.cursors.get("live_cursor").unwrap().position, 3);

    execute_fetch_forward(&mut writer, &mut session, "live_cursor", None).unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'C']
    );
    assert_eq!(messages[1].1, b"FETCH 0\0".to_vec());
}

#[test]
fn extended_cursor_close_missing_name_errors_without_clearing_live_cursors() {
    let mut session = Session::default();
    session.cursors.insert(
        "live_cursor".to_string(),
        Cursor {
            columns: vec![int4_column("id")],
            rows: vec![vec![Some("1".to_string())]],
            position: 0,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(&mut writer, &mut session, "CLOSE missing_cursor", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(session.cursors.contains_key("live_cursor"));

    execute_statement(&mut writer, &mut session, "CLOSE live_cursor", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(!session.cursors.contains_key("live_cursor"));
}

#[test]
fn transaction_end_closes_session_local_cursors() {
    let mut session = Session::default();
    session.cursors.insert(
        "commit_cursor".to_string(),
        Cursor {
            columns: vec![int4_column("id")],
            rows: vec![vec![Some("1".to_string())]],
            position: 0,
        },
    );
    session.cursors.insert(
        "rollback_cursor".to_string(),
        Cursor {
            columns: vec![int4_column("id")],
            rows: vec![vec![Some("2".to_string())]],
            position: 0,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(&mut writer, &mut session, "COMMIT AND CHAIN", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(session.in_transaction);
    assert!(session.cursors.is_empty());

    session.cursors.insert(
        "rollback_cursor".to_string(),
        Cursor {
            columns: vec![int4_column("id")],
            rows: vec![vec![Some("2".to_string())]],
            position: 0,
        },
    );
    execute_statement(&mut writer, &mut session, "ROLLBACK", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(!session.in_transaction);
    assert!(session.cursors.is_empty());
}

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
