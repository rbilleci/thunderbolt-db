//! Relational SQL parser-facade compatibility coverage.

use super::*;

#[test]
fn parses_minimal_relational_sql_subset() {
    assert_eq!(
        parse_command("CREATE TABLE people (id INT, name TEXT)").unwrap(),
        Command::CreateTable(CreateTable {
            table: "people".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
                ColumnDef {
                    name: "name".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: None,
                },
            ],
            primary_key: None,
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );

    assert_eq!(
        parse_command("CREATE TABLE keyed_people (id INT PRIMARY KEY, name TEXT)").unwrap(),
        Command::CreateTable(CreateTable {
            table: "keyed_people".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
                ColumnDef {
                    name: "name".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: None,
                },
            ],
            primary_key: Some(PrimaryKey {
                name: None,
                column: "id".to_string(),
                columns: vec!["id".to_string()],
            }),
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );

    assert_eq!(
        parse_command(
            "CREATE TABLE unique_people (id INT, name TEXT UNIQUE, CONSTRAINT unique_people_id_key UNIQUE (id))"
        )
        .unwrap(),
        Command::CreateTable(CreateTable {
            table: "unique_people".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
default: None,
                },
                ColumnDef {
                    name: "name".to_string(),
                    ty: SqlType::Text,
                    domain: None,
default: None,
                },
            ],
            primary_key: None,
            unique_constraints: vec![
                UniqueConstraint {
                    name: None,
                    column: "name".to_string(),
                    columns: vec!["name".to_string()],
                },
                UniqueConstraint {
                    name: Some("unique_people_id_key".to_string()),
                    column: "id".to_string(),
                    columns: vec!["id".to_string()],
                },
            ],
            check_constraints: Vec::new(),
        })
    );

    assert_eq!(
        parse_command(
            "CREATE TABLE check_people (id INT, name TEXT, CONSTRAINT check_people_id_positive CHECK (id > 0), CHECK (name = 'Ada'))"
        )
        .unwrap(),
        Command::CreateTable(CreateTable {
            table: "check_people".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
                ColumnDef {
                    name: "name".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: None,
                },
            ],
            primary_key: None,
            unique_constraints: Vec::new(),
            check_constraints: vec![
                CheckConstraint {
                    name: Some("check_people_id_positive".to_string()),
                    filter: SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gt,
                        value: SqlValue::Int4(0),
                    },
                },
                CheckConstraint {
                    name: None,
                    filter: SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                },
            ],
        })
    );
    assert!(matches!(
        parse_command("CREATE TABLE bad_check_people (id INT, CHECK (missing > 0))"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE TABLE bad_check_people (id INT, CHECK (id BETWEEN 1 AND 3))"),
        Err(ParseError::InvalidRelationalSql)
    ));

    assert_eq!(
        parse_command("ALTER TABLE ONLY public.keyed_people ADD CONSTRAINT keyed_people_pkey PRIMARY KEY (id)").unwrap(),
        Command::AddPrimaryKey(AddPrimaryKey {
            table: "keyed_people".to_string(),
            name: "keyed_people_pkey".to_string(),
            column: "id".to_string(),
            columns: vec!["id".to_string()],
        })
    );
    assert_eq!(
        parse_command("ALTER TABLE ONLY public.keyed_people ADD CONSTRAINT keyed_people_name_key UNIQUE (name)").unwrap(),
        Command::AddUniqueConstraint(AddUniqueConstraint {
            table: "keyed_people".to_string(),
            name: "keyed_people_name_key".to_string(),
            column: "name".to_string(),
            columns: vec!["name".to_string()],
        })
    );
    assert_eq!(
        parse_command(
            "ALTER TABLE ONLY public.keyed_people ADD CONSTRAINT keyed_people_id_positive CHECK (id > 0)"
        )
        .unwrap(),
        Command::AddCheckConstraint(AddCheckConstraint {
            table: "keyed_people".to_string(),
            name: "keyed_people_id_positive".to_string(),
            filter: SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gt,
                value: SqlValue::Int4(0),
            },
        })
    );
    assert!(matches!(
        parse_command(
            "ALTER TABLE ONLY public.keyed_people ADD CONSTRAINT keyed_people_id_between CHECK (id BETWEEN 1 AND 3)"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command(
            "ALTER TABLE ONLY public.orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES public.customers(id)"
        )
        .unwrap(),
        Command::AddForeignKey(AddForeignKey {
            table: "orders".to_string(),
            name: "orders_customer_fk".to_string(),
            column: "customer_id".to_string(),
            referenced_table: "customers".to_string(),
            referenced_column: "id".to_string(),
        })
    );
    assert!(matches!(
        parse_command(
            "ALTER TABLE orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id, tenant_id) REFERENCES customers(id, tenant_id)"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command(
            "ALTER TABLE orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES private.customers(id)"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command(
            "ALTER TABLE orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command(
            "ALTER TABLE IF EXISTS ONLY public.keyed_people DROP CONSTRAINT IF EXISTS keyed_people_name_key"
        )
        .unwrap(),
        Command::DropConstraint(DropConstraint {
            table: "keyed_people".to_string(),
            name: "keyed_people_name_key".to_string(),
            table_if_exists: true,
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("ALTER TABLE ONLY keyed_people DROP CONSTRAINT keyed_people_pkey").unwrap(),
        Command::DropConstraint(DropConstraint {
            table: "keyed_people".to_string(),
            name: "keyed_people_pkey".to_string(),
            table_if_exists: false,
            if_exists: false,
        })
    );
    assert!(parse_command(
        "ALTER TABLE ONLY public.keyed_people DROP CONSTRAINT keyed_people_pkey CASCADE"
    )
    .is_err());
    assert_eq!(
        parse_command("ALTER TABLE IF EXISTS ONLY public.keyed_people RENAME TO archived_people")
            .unwrap(),
        Command::RenameTable(RenameTable {
            old_name: "keyed_people".to_string(),
            new_name: "archived_people".to_string(),
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("ALTER TABLE public.keyed_people RENAME TO renamed_people").unwrap(),
        Command::RenameTable(RenameTable {
            old_name: "keyed_people".to_string(),
            new_name: "renamed_people".to_string(),
            if_exists: false,
        })
    );
    assert!(
        parse_command("ALTER TABLE public.keyed_people RENAME TO public.renamed_people").is_err()
    );
    assert!(
        parse_command("ALTER TABLE public.keyed_people RENAME TO renamed_people CASCADE").is_err()
    );
    assert_eq!(
        parse_command("ALTER TABLE ONLY public.keyed_people RENAME COLUMN name TO display_name")
            .unwrap(),
        Command::RenameColumn(RenameColumn {
            table: "keyed_people".to_string(),
            old_name: "name".to_string(),
            new_name: "display_name".to_string(),
        })
    );
    assert_eq!(
        parse_command("ALTER TABLE public.keyed_people RENAME id TO person_id").unwrap(),
        Command::RenameColumn(RenameColumn {
            table: "keyed_people".to_string(),
            old_name: "id".to_string(),
            new_name: "person_id".to_string(),
        })
    );
    assert!(parse_command(
        "ALTER TABLE public.keyed_people RENAME COLUMN name TO display_name CASCADE"
    )
    .is_err());
    assert_eq!(
        parse_command(
            "ALTER TABLE IF EXISTS ONLY public.keyed_people RENAME CONSTRAINT keyed_people_pkey TO keyed_people_id_pkey"
        )
        .unwrap(),
        Command::RenameConstraint(RenameConstraint {
            table: "keyed_people".to_string(),
            old_name: "keyed_people_pkey".to_string(),
            new_name: "keyed_people_id_pkey".to_string(),
            table_if_exists: true,
        })
    );
    assert!(parse_command(
        "ALTER TABLE public.keyed_people RENAME CONSTRAINT keyed_people_pkey TO keyed_people_id_pkey CASCADE"
    )
    .is_err());

    assert_eq!(
        parse_command("COMMENT ON DATABASE postgres IS 'primary database'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Database {
                database: "postgres".to_string(),
            },
            comment: Some("primary database".to_string()),
        })
    );

    assert_eq!(
        parse_command("COMMENT ON SCHEMA public IS 'application schema'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Schema {
                schema: "public".to_string(),
            },
            comment: Some("application schema".to_string()),
        })
    );

    assert_eq!(
        parse_command("COMMENT ON ROLE postgres IS 'bootstrap role'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Role {
                role: "postgres".to_string(),
            },
            comment: Some("bootstrap role".to_string()),
        })
    );

    assert_eq!(
        parse_command("COMMENT ON TABLESPACE pg_default IS 'default storage'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Tablespace {
                tablespace: "pg_default".to_string(),
            },
            comment: Some("default storage".to_string()),
        })
    );

    assert_eq!(
        parse_command("COMMENT ON TABLE public.people IS 'lookup people'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Table {
                table: "people".to_string(),
            },
            comment: Some("lookup people".to_string()),
        })
    );

    assert_eq!(
        parse_command("COMMENT ON COLUMN public.people.name IS 'display name'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Column {
                table: "people".to_string(),
                column: "name".to_string(),
            },
            comment: Some("display name".to_string()),
        })
    );

    assert_eq!(
        parse_command("COMMENT ON COLUMN public.people.name IS NULL").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Column {
                table: "people".to_string(),
                column: "name".to_string(),
            },
            comment: None,
        })
    );

    assert_eq!(
        parse_command("COMMENT ON INDEX public.people_name_idx IS 'lookup index'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Index {
                index: "people_name_idx".to_string(),
            },
            comment: Some("lookup index".to_string()),
        })
    );

    assert_eq!(
        parse_command("COMMENT ON VIEW public.active_people IS 'active people view'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::View {
                view: "active_people".to_string(),
            },
            comment: Some("active people view".to_string()),
        })
    );

    assert_eq!(
        parse_command("COMMENT ON CONSTRAINT people_pkey ON public.people IS 'row identity'")
            .unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Constraint {
                table: "people".to_string(),
                constraint: "people_pkey".to_string(),
            },
            comment: Some("row identity".to_string()),
        })
    );

    assert_eq!(
        parse_command("CREATE TABLE typed_people (id INT4, owner pg_catalog.text)").unwrap(),
        Command::CreateTable(CreateTable {
            table: "typed_people".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
                ColumnDef {
                    name: "owner".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: None,
                },
            ],
            primary_key: None,
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );

    assert_eq!(
        parse_command("CREATE TABLE public.dump_people (id integer, name text)").unwrap(),
        Command::CreateTable(CreateTable {
            table: "dump_people".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
                ColumnDef {
                    name: "name".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: None,
                },
            ],
            primary_key: None,
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );
    assert!(matches!(
        parse_command("CREATE TABLE private.dump_people (id integer)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command(
            "CREATE TABLE default_people (id INT DEFAULT 7, name TEXT DEFAULT 'Ada''s'::text)"
        )
        .unwrap(),
        Command::CreateTable(CreateTable {
            table: "default_people".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: Some(ColumnDefault::Literal(SqlValue::Int4(7))),
                },
                ColumnDef {
                    name: "name".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: Some(ColumnDefault::Literal(SqlValue::Text("Ada's".to_string()))),
                },
            ],
            primary_key: None,
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );
    assert!(matches!(
        parse_command("CREATE TABLE invalid_default (id INT DEFAULT 'bad'::text)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("CREATE TABLE serial_people (id SERIAL PRIMARY KEY, name TEXT)").unwrap(),
        Command::CreateTable(CreateTable {
            table: "serial_people".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: Some(ColumnDefault::SequenceNextVal {
                        sequence: "serial_people_id_seq".to_string(),
                        create_if_missing: true,
                    }),
                },
                ColumnDef {
                    name: "name".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: None,
                },
            ],
            primary_key: Some(PrimaryKey {
                name: None,
                column: "id".to_string(),
                columns: vec!["id".to_string()],
            }),
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );
    assert_eq!(
        parse_command(
            "CREATE TABLE seq_default_people (id INT DEFAULT nextval('public.people_seq'::regclass))"
        )
        .unwrap(),
        Command::CreateTable(CreateTable {
            table: "seq_default_people".to_string(),
            columns: vec![ColumnDef {
                name: "id".to_string(),
                ty: SqlType::Int4,
                domain: None,
default: Some(ColumnDefault::SequenceNextVal {
                    sequence: "people_seq".to_string(),
                    create_if_missing: false,
                }),
            }],
            primary_key: None,
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );
    assert!(matches!(
        parse_command("CREATE TABLE invalid_serial (id TEXT DEFAULT nextval('people_seq'))"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("CREATE TABLE type_named_columns (integer_col integer, text_col text)")
            .unwrap(),
        Command::CreateTable(CreateTable {
            table: "type_named_columns".to_string(),
            columns: vec![
                ColumnDef {
                    name: "integer_col".to_string(),
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
                ColumnDef {
                    name: "text_col".to_string(),
                    ty: SqlType::Text,
                    domain: None,
                    default: None,
                },
            ],
            primary_key: None,
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );
    assert_eq!(
        parse_command(
            "ALTER TABLE ONLY public.default_people ALTER COLUMN name SET DEFAULT 'Grace'::text"
        )
        .unwrap(),
        Command::AlterColumnDefault(AlterColumnDefault {
            table: "default_people".to_string(),
            column: "name".to_string(),
            default: Some(ColumnDefault::Literal(SqlValue::Text("Grace".to_string()))),
        })
    );
    assert_eq!(
        parse_command("ALTER TABLE ONLY public.default_people ALTER COLUMN id SET DEFAULT nextval('public.default_people_id_seq'::regclass)").unwrap(),
        Command::AlterColumnDefault(AlterColumnDefault {
            table: "default_people".to_string(),
            column: "id".to_string(),
            default: Some(ColumnDefault::SequenceNextVal {
                sequence: "default_people_id_seq".to_string(),
                create_if_missing: false,
            }),
        })
    );
    assert_eq!(
        parse_command("ALTER TABLE public.default_people ALTER name DROP DEFAULT").unwrap(),
        Command::AlterColumnDefault(AlterColumnDefault {
            table: "default_people".to_string(),
            column: "name".to_string(),
            default: None,
        })
    );
    assert_eq!(
        parse_command(
            "ALTER TABLE ONLY public.default_people ADD COLUMN tag TEXT DEFAULT 'new'::text"
        )
        .unwrap(),
        Command::AddColumn(AddColumn {
            table: "default_people".to_string(),
            column: ColumnDef {
                name: "tag".to_string(),
                ty: SqlType::Text,
                domain: None,
                default: Some(ColumnDefault::Literal(SqlValue::Text("new".to_string()))),
            },
        })
    );
    assert_eq!(
        parse_command("ALTER TABLE public.default_people ADD bucket INT DEFAULT 4").unwrap(),
        Command::AddColumn(AddColumn {
            table: "default_people".to_string(),
            column: ColumnDef {
                name: "bucket".to_string(),
                ty: SqlType::Int4,
                domain: None,
                default: Some(ColumnDefault::Literal(SqlValue::Int4(4))),
            },
        })
    );
    assert_eq!(
        parse_command(
            "ALTER TABLE public.default_people ADD bucket INT DEFAULT nextval('public.default_bucket_seq'::regclass)"
        )
        .unwrap(),
        Command::AddColumn(AddColumn {
            table: "default_people".to_string(),
            column: ColumnDef {
                name: "bucket".to_string(),
                ty: SqlType::Int4,
                domain: None,
default: Some(ColumnDefault::SequenceNextVal {
                    sequence: "default_bucket_seq".to_string(),
                    create_if_missing: false,
                }),
            },
        })
    );
    assert!(matches!(
        parse_command("ALTER TABLE private.default_people ADD COLUMN tag TEXT DEFAULT 'bad'::text"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("ALTER TABLE ONLY public.default_people DROP COLUMN tag").unwrap(),
        Command::DropColumn(DropColumn {
            table: "default_people".to_string(),
            column: "tag".to_string(),
        })
    );
    assert_eq!(
        parse_command("ALTER TABLE default_people DROP bucket").unwrap(),
        Command::DropColumn(DropColumn {
            table: "default_people".to_string(),
            column: "bucket".to_string(),
        })
    );
    assert!(matches!(
        parse_command("ALTER TABLE default_people DROP COLUMN tag CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER TABLE default_people DROP COLUMN tag, bucket"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("CREATE VIEW public.active_people AS SELECT id, name FROM people WHERE id > 1 ORDER BY id LIMIT 5").unwrap(),
        Command::CreateView(CreateView {
            name: "active_people".to_string(),
            query: Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec![
                    "id".to_string(),
                    "name".to_string(),
                ]),
                group_by: None,
                having_groups: Vec::new(),
                filter: Some(SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gt,
                    value: SqlValue::Int4(1),
                }),
                filters: vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gt,
                    value: SqlValue::Int4(1),
                }],
                filter_groups: vec![vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gt,
                    value: SqlValue::Int4(1),
                }]],
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: Some(5),
                offset: None,
            },
            definition: "SELECT id, name FROM people WHERE id > 1 ORDER BY id LIMIT 5"
                .to_string(),
            or_replace: false,
        })
    );
    assert_eq!(
        parse_command("CREATE OR REPLACE VIEW active_people AS SELECT * FROM people").unwrap(),
        Command::CreateView(CreateView {
            name: "active_people".to_string(),
            query: Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::All,
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: Vec::new(),
                limit: None,
                offset: None,
            },
            definition: "SELECT * FROM people".to_string(),
            or_replace: true,
        })
    );
    assert_eq!(
        parse_command("DROP VIEW public.active_people").unwrap(),
        Command::DropView(DropView {
            names: vec!["active_people".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(
        parse_command("DROP VIEW IF EXISTS active_people").unwrap(),
        Command::DropView(DropView {
            names: vec!["active_people".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("CREATE SCHEMA IF NOT EXISTS public").unwrap(),
        Command::CreateSchema(CreateSchema {
            name: "public".to_string(),
            if_not_exists: true,
        })
    );
    assert_eq!(
        parse_command("DROP SCHEMA IF EXISTS public").unwrap(),
        Command::DropSchema(DropSchema {
            name: "public".to_string(),
            if_exists: true,
        })
    );
    assert!(parse_command("DROP SCHEMA public CASCADE").is_err());
    assert_eq!(
        parse_command("DROP VIEW public.a, public.b").unwrap(),
        Command::DropView(DropView {
            names: vec!["a".to_string(), "b".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(
        parse_command("ALTER VIEW public.active_people RENAME TO renamed_people").unwrap(),
        Command::RenameView(RenameView {
            old_name: "active_people".to_string(),
            new_name: "renamed_people".to_string(),
        })
    );
    assert!(matches!(
        parse_command("DROP VIEW active_people CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER VIEW active_people RENAME TO renamed_people CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("ALTER MATERIALIZED VIEW active_people RENAME TO renamed_people").unwrap(),
        Command::RenameMaterializedView(RenameMaterializedView {
            old_name: "active_people".to_string(),
            new_name: "renamed_people".to_string(),
        })
    );
    assert!(matches!(
        parse_command("ALTER VIEW public.active_people RENAME TO public.renamed_people"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("DROP MATERIALIZED VIEW active_people").unwrap(),
        Command::DropMaterializedView(DropMaterializedView {
            names: vec!["active_people".to_string()],
            if_exists: false,
        })
    );

    assert_eq!(
        parse_command("DROP TABLE public.people").unwrap(),
        Command::DropTable(DropTable {
            names: vec!["people".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(
        parse_command("DROP TABLE IF EXISTS people").unwrap(),
        Command::DropTable(DropTable {
            names: vec!["people".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("DROP TABLE public.a, public.b").unwrap(),
        Command::DropTable(DropTable {
            names: vec!["a".to_string(), "b".to_string()],
            if_exists: false,
        })
    );
    assert!(matches!(
        parse_command("DROP TABLE people CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP TABLE private.people"),
        Err(ParseError::InvalidRelationalSql)
    ));

    assert_eq!(
        parse_command("TRUNCATE TABLE ONLY public.people").unwrap(),
        Command::TruncateTable(TruncateTable {
            name: "people".to_string(),
            restart_identity: false,
        })
    );
    assert_eq!(
        parse_command("TRUNCATE people").unwrap(),
        Command::TruncateTable(TruncateTable {
            name: "people".to_string(),
            restart_identity: false,
        })
    );
    assert_eq!(
        parse_command("TRUNCATE TABLE people RESTART IDENTITY").unwrap(),
        Command::TruncateTable(TruncateTable {
            name: "people".to_string(),
            restart_identity: true,
        })
    );
    assert!(matches!(
        parse_command("TRUNCATE TABLE people CONTINUE IDENTITY"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("TRUNCATE TABLE people CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("TRUNCATE TABLE public.a, public.b"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("TRUNCATE TABLE private.people"),
        Err(ParseError::InvalidRelationalSql)
    ));

    assert_eq!(
        parse_command("CREATE PUBLICATION app_pub FOR TABLE public.people, accounts").unwrap(),
        Command::CreatePublication(CreatePublication {
            name: "app_pub".to_string(),
            target: PublicationTarget::Tables(vec!["people".to_string(), "accounts".to_string(),]),
        })
    );
    assert_eq!(
        parse_command("CREATE PUBLICATION all_pub FOR ALL TABLES").unwrap(),
        Command::CreatePublication(CreatePublication {
            name: "all_pub".to_string(),
            target: PublicationTarget::AllTables,
        })
    );
    assert_eq!(
        parse_command("DROP PUBLICATION IF EXISTS app_pub, all_pub").unwrap(),
        Command::DropPublication(DropPublication {
            names: vec!["app_pub".to_string(), "all_pub".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("COMMENT ON PUBLICATION app_pub IS 'app publication'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Publication {
                publication: "app_pub".to_string(),
            },
            comment: Some("app publication".to_string()),
        })
    );
    assert!(matches!(
        parse_command("CREATE PUBLICATION app_pub FOR TABLE private.people"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE PUBLICATION app_pub FOR TABLE people WITH (publish = 'insert')"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP PUBLICATION app_pub CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command(
            "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost dbname=postgres' PUBLICATION app_pub, all_pub WITH (connect = false, enabled = false)"
        )
        .unwrap(),
        Command::CreateSubscription(CreateSubscription {
            name: "app_sub".to_string(),
            connection: "host=localhost dbname=postgres".to_string(),
            publications: vec!["app_pub".to_string(), "all_pub".to_string()],
        })
    );
    assert_eq!(
        parse_command("DROP SUBSCRIPTION IF EXISTS app_sub, stale_sub").unwrap(),
        Command::DropSubscription(DropSubscription {
            names: vec!["app_sub".to_string(), "stale_sub".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("COMMENT ON SUBSCRIPTION app_sub IS NULL").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Subscription {
                subscription: "app_sub".to_string(),
            },
            comment: None,
        })
    );
    assert!(matches!(
        parse_command(
            "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost' PUBLICATION app_pub"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command(
            "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost' PUBLICATION app_pub WITH (connect = true, enabled = false)"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command(
            "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost' PUBLICATION app_pub WITH (connect = false, enabled = true)"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP SUBSCRIPTION app_sub CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));

    assert_eq!(
        parse_command("GRANT SELECT, INSERT ON TABLE public.people TO PUBLIC").unwrap(),
        Command::GrantTable(GrantTable {
            relation: "people".to_string(),
            kind: AclRelationKind::Table,
            grantee: "public".to_string(),
            privileges: vec![TablePrivilege::Select, TablePrivilege::Insert],
        })
    );
    assert_eq!(
        parse_command("GRANT ALL PRIVILEGES ON people TO postgres").unwrap(),
        Command::GrantTable(GrantTable {
            relation: "people".to_string(),
            kind: AclRelationKind::Relation,
            grantee: "postgres".to_string(),
            privileges: vec![
                TablePrivilege::Select,
                TablePrivilege::Insert,
                TablePrivilege::Update,
                TablePrivilege::Delete,
            ],
        })
    );
    assert_eq!(
        parse_command("REVOKE UPDATE, DELETE ON TABLE people FROM PUBLIC").unwrap(),
        Command::RevokeTable(RevokeTable {
            relation: "people".to_string(),
            kind: AclRelationKind::Table,
            grantee: "public".to_string(),
            privileges: vec![TablePrivilege::Update, TablePrivilege::Delete],
        })
    );
    assert_eq!(
        parse_command("GRANT SELECT ON VIEW public.people_view TO PUBLIC").unwrap(),
        Command::GrantTable(GrantTable {
            relation: "people_view".to_string(),
            kind: AclRelationKind::View,
            grantee: "public".to_string(),
            privileges: vec![TablePrivilege::Select],
        })
    );
    assert_eq!(
        parse_command("REVOKE SELECT ON MATERIALIZED VIEW people_mv FROM PUBLIC").unwrap(),
        Command::RevokeTable(RevokeTable {
            relation: "people_mv".to_string(),
            kind: AclRelationKind::MaterializedView,
            grantee: "public".to_string(),
            privileges: vec![TablePrivilege::Select],
        })
    );
    assert_eq!(
        parse_command("GRANT SELECT, UPDATE ON SEQUENCE people_id_seq TO postgres").unwrap(),
        Command::GrantTable(GrantTable {
            relation: "people_id_seq".to_string(),
            kind: AclRelationKind::Sequence,
            grantee: "postgres".to_string(),
            privileges: vec![TablePrivilege::Select, TablePrivilege::Update],
        })
    );
    assert!(matches!(
        parse_command("GRANT SELECT (id) ON people TO PUBLIC"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("GRANT SELECT ON TABLE private.people TO PUBLIC"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("GRANT SELECT ON people TO app_reader").unwrap(),
        Command::GrantTable(GrantTable {
            relation: "people".to_string(),
            kind: AclRelationKind::Relation,
            grantee: "app_reader".to_string(),
            privileges: vec![TablePrivilege::Select],
        })
    );
    assert_eq!(
        parse_command("CREATE ROLE app_reader WITH LOGIN").unwrap(),
        Command::CreateRole(CreateRole {
            name: "app_reader".to_string(),
            login: true,
        })
    );
    assert_eq!(
        parse_command("CREATE USER app_writer").unwrap(),
        Command::CreateRole(CreateRole {
            name: "app_writer".to_string(),
            login: true,
        })
    );
    assert_eq!(
        parse_command("CREATE ROLE app_batch NOLOGIN").unwrap(),
        Command::CreateRole(CreateRole {
            name: "app_batch".to_string(),
            login: false,
        })
    );
    assert_eq!(
        parse_command("DROP ROLE IF EXISTS app_reader, app_writer").unwrap(),
        Command::DropRole(DropRole {
            names: vec!["app_reader".to_string(), "app_writer".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("ALTER ROLE app_reader RENAME TO app_analyst").unwrap(),
        Command::RenameRole(RenameRole {
            old_name: "app_reader".to_string(),
            new_name: "app_analyst".to_string(),
        })
    );
    assert_eq!(
        parse_command("DROP USER app_batch").unwrap(),
        Command::DropRole(DropRole {
            names: vec!["app_batch".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(
        parse_command("CREATE DATABASE appdb").unwrap(),
        Command::CreateDatabase(CreateDatabase {
            name: "appdb".to_string(),
        })
    );
    assert_eq!(
        parse_command("CREATE DATABASE \"App DB\"").unwrap(),
        Command::CreateDatabase(CreateDatabase {
            name: "App DB".to_string(),
        })
    );
    assert_eq!(
        parse_command("DROP DATABASE IF EXISTS appdb, stale_db").unwrap(),
        Command::DropDatabase(DropDatabase {
            names: vec!["appdb".to_string(), "stale_db".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("ALTER DATABASE appdb RENAME TO appdb_archive").unwrap(),
        Command::RenameDatabase(RenameDatabase {
            old_name: "appdb".to_string(),
            new_name: "appdb_archive".to_string(),
        })
    );
    assert_eq!(
        parse_command("CREATE TABLESPACE appspace LOCATION '/tmp/gpu-db-appspace'").unwrap(),
        Command::CreateTablespace(CreateTablespace {
            name: "appspace".to_string(),
            location: "/tmp/gpu-db-appspace".to_string(),
        })
    );
    assert_eq!(
        parse_command("CREATE TABLESPACE \"App Space\" LOCATION '/tmp/app space'").unwrap(),
        Command::CreateTablespace(CreateTablespace {
            name: "App Space".to_string(),
            location: "/tmp/app space".to_string(),
        })
    );
    assert_eq!(
        parse_command("CREATE TABLESPACE appspace OWNER postgres LOCATION '/tmp/appspace'")
            .unwrap(),
        Command::CreateTablespace(CreateTablespace {
            name: "appspace".to_string(),
            location: "/tmp/appspace".to_string(),
        })
    );
    assert_eq!(
        parse_command("DROP TABLESPACE IF EXISTS appspace, stale_space").unwrap(),
        Command::DropTablespace(DropTablespace {
            names: vec!["appspace".to_string(), "stale_space".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("ALTER TABLESPACE appspace RENAME TO appspace_fast").unwrap(),
        Command::RenameTablespace(RenameTablespace {
            old_name: "appspace".to_string(),
            new_name: "appspace_fast".to_string(),
        })
    );
    assert!(matches!(
        parse_command("CREATE DATABASE appdb OWNER postgres"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP DATABASE appdb WITH (FORCE)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER DATABASE appdb OWNER TO postgres"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER DATABASE appdb RENAME TO appdb_archive SET TABLESPACE pg_default"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE TABLESPACE appspace OWNER app_owner LOCATION '/tmp/appspace'"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE TABLESPACE appspace"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP TABLESPACE appspace CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER TABLESPACE appspace OWNER TO postgres"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER TABLESPACE appspace RENAME TO public.appspace"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE ROLE app_reader PASSWORD 'secret'"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE ROLE app_reader LOGIN NOLOGIN"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER ROLE app_reader RENAME TO app_analyst WITH LOGIN"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP ROLE app_reader CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("GRANT SELECT ON people TO PUBLIC WITH GRANT OPTION"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("GRANT USAGE, CREATE ON SCHEMA public TO PUBLIC").unwrap(),
        Command::GrantSchema(SchemaPrivileges {
            schema: "public".to_string(),
            grantee: "public".to_string(),
            privileges: vec![SchemaPrivilege::Usage, SchemaPrivilege::Create],
        })
    );
    assert_eq!(
        parse_command("REVOKE CREATE ON SCHEMA public FROM postgres").unwrap(),
        Command::RevokeSchema(SchemaPrivileges {
            schema: "public".to_string(),
            grantee: "postgres".to_string(),
            privileges: vec![SchemaPrivilege::Create],
        })
    );
    assert!(matches!(
        parse_command("GRANT USAGE ON SCHEMA private TO PUBLIC"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("GRANT USAGE ON SCHEMA public TO app_reader").unwrap(),
        Command::GrantSchema(SchemaPrivileges {
            schema: "public".to_string(),
            grantee: "app_reader".to_string(),
            privileges: vec![SchemaPrivilege::Usage],
        })
    );
    assert!(matches!(
        parse_command("GRANT USAGE ON SCHEMA public TO PUBLIC WITH GRANT OPTION"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command(
            "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT, INSERT ON TABLES TO PUBLIC"
        )
        .unwrap(),
        Command::GrantDefaultTablePrivileges(DefaultTablePrivileges {
            grantee: "public".to_string(),
            privileges: vec![TablePrivilege::Select, TablePrivilege::Insert],
        })
    );
    assert_eq!(
        parse_command("ALTER DEFAULT PRIVILEGES REVOKE INSERT ON TABLES FROM PUBLIC").unwrap(),
        Command::RevokeDefaultTablePrivileges(DefaultTablePrivileges {
            grantee: "public".to_string(),
            privileges: vec![TablePrivilege::Insert],
        })
    );
    assert!(matches!(
        parse_command(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA private GRANT SELECT ON TABLES TO PUBLIC"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER DEFAULT PRIVILEGES GRANT SELECT ON SEQUENCES TO PUBLIC"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO app_reader").unwrap(),
        Command::GrantDefaultTablePrivileges(DefaultTablePrivileges {
            grantee: "app_reader".to_string(),
            privileges: vec![TablePrivilege::Select],
        })
    );
    assert_eq!(
        parse_command("GRANT CONNECT, TEMPORARY ON DATABASE appdb TO app_reader").unwrap(),
        Command::GrantDatabase(DatabasePrivileges {
            database: "appdb".to_string(),
            grantee: "app_reader".to_string(),
            privileges: vec![DatabasePrivilege::Connect, DatabasePrivilege::Temporary],
        })
    );
    assert_eq!(
        parse_command("REVOKE TEMP ON DATABASE appdb FROM PUBLIC").unwrap(),
        Command::RevokeDatabase(DatabasePrivileges {
            database: "appdb".to_string(),
            grantee: "public".to_string(),
            privileges: vec![DatabasePrivilege::Temporary],
        })
    );
    assert_eq!(
        parse_command("GRANT ALL PRIVILEGES ON TABLESPACE appspace TO postgres").unwrap(),
        Command::GrantTablespace(TablespacePrivileges {
            tablespace: "appspace".to_string(),
            grantee: "postgres".to_string(),
            privileges: vec![TablespacePrivilege::Create],
        })
    );
    assert_eq!(
        parse_command("REVOKE CREATE ON TABLESPACE appspace FROM app_reader").unwrap(),
        Command::RevokeTablespace(TablespacePrivileges {
            tablespace: "appspace".to_string(),
            grantee: "app_reader".to_string(),
            privileges: vec![TablespacePrivilege::Create],
        })
    );
    assert_eq!(
        parse_command("GRANT EXECUTE ON FUNCTION public.answer() TO app_reader").unwrap(),
        Command::GrantFunction(FunctionPrivileges {
            function: "answer".to_string(),
            grantee: "app_reader".to_string(),
            privileges: vec![FunctionPrivilege::Execute],
        })
    );
    assert_eq!(
        parse_command("REVOKE ALL PRIVILEGES ON FUNCTION answer() FROM PUBLIC").unwrap(),
        Command::RevokeFunction(FunctionPrivileges {
            function: "answer".to_string(),
            grantee: "public".to_string(),
            privileges: vec![FunctionPrivilege::Execute],
        })
    );
    assert!(matches!(
        parse_command("GRANT SELECT ON FUNCTION answer() TO PUBLIC"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("GRANT EXECUTE ON FUNCTION answer(int4) TO PUBLIC"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("GRANT SELECT ON DATABASE appdb TO PUBLIC"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("GRANT CREATE ON TABLESPACE appspace TO PUBLIC WITH GRANT OPTION"),
        Err(ParseError::InvalidRelationalSql)
    ));

    assert_eq!(
        parse_command("CREATE INDEX people_name_idx ON public.people (name)").unwrap(),
        Command::CreateIndex(CreateIndex {
            name: "people_name_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            columns: vec!["name".to_string()],
            unique: false,
        })
    );
    assert_eq!(
        parse_command("CREATE INDEX people_name_idx ON public.people USING btree (name)").unwrap(),
        Command::CreateIndex(CreateIndex {
            name: "people_name_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            columns: vec!["name".to_string()],
            unique: false,
        })
    );
    assert_eq!(
        parse_command("CREATE UNIQUE INDEX people_name_idx ON public.people USING btree (name)")
            .unwrap(),
        Command::CreateIndex(CreateIndex {
            name: "people_name_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            columns: vec!["name".to_string()],
            unique: true,
        })
    );
    assert!(matches!(
        parse_command("CREATE INDEX people_name_idx ON people USING hash (name)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE INDEX people_multi_idx ON people (id, name)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("DROP INDEX public.people_name_idx").unwrap(),
        Command::DropIndex(DropIndex {
            names: vec!["people_name_idx".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(
        parse_command("DROP INDEX IF EXISTS people_name_idx").unwrap(),
        Command::DropIndex(DropIndex {
            names: vec!["people_name_idx".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("DROP INDEX public.a, public.b").unwrap(),
        Command::DropIndex(DropIndex {
            names: vec!["a".to_string(), "b".to_string()],
            if_exists: false,
        })
    );
    assert!(matches!(
        parse_command("DROP INDEX CONCURRENTLY people_name_idx"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP INDEX people_name_idx CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("ALTER INDEX public.people_name_idx RENAME TO people_lookup_idx").unwrap(),
        Command::RenameIndex(RenameIndex {
            old_name: "people_name_idx".to_string(),
            new_name: "people_lookup_idx".to_string(),
        })
    );
    assert!(matches!(
        parse_command("ALTER INDEX IF EXISTS people_name_idx RENAME TO people_lookup_idx"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER INDEX people_name_idx RENAME TO public.people_lookup_idx"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER INDEX people_name_idx RENAME TO people_lookup_idx CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));

    assert_eq!(
        parse_command("INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')").unwrap(),
        Command::Insert(Insert {
            table: "people".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            ],
        })
    );

    assert_eq!(
        parse_command("INSERT INTO people (id, name) VALUES (1, 'O''Brien')").unwrap(),
        Command::Insert(Insert {
            table: "people".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![vec![
                SqlValue::Int4(1),
                SqlValue::Text("O'Brien".to_string())
            ]],
        })
    );

    assert_eq!(
        parse_command("INSERT INTO people (id, name) VALUES (-1, 'Minus'), (0, 'Zero')").unwrap(),
        Command::Insert(Insert {
            table: "people".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![
                vec![SqlValue::Int4(-1), SqlValue::Text("Minus".to_string())],
                vec![SqlValue::Int4(0), SqlValue::Text("Zero".to_string())],
            ],
        })
    );

    assert_eq!(
        parse_command("INSERT INTO public.people (id, name) VALUES (3, 'Grace')").unwrap(),
        Command::Insert(Insert {
            table: "people".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())]],
        })
    );

    assert_eq!(
        parse_command("INSERT INTO public.people VALUES (4, 'Katherine'), (5, 'Mary')").unwrap(),
        Command::Insert(Insert {
            table: "people".to_string(),
            columns: Vec::new(),
            rows: vec![
                vec![SqlValue::Int4(4), SqlValue::Text("Katherine".to_string())],
                vec![SqlValue::Int4(5), SqlValue::Text("Mary".to_string())],
            ],
        })
    );

    assert_eq!(
        parse_command("DELETE FROM public.people WHERE id = 1 OR name LIKE 'Ada%'").unwrap(),
        Command::Delete(Delete {
            table: "people".to_string(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }],
            filter_groups: vec![
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("Ada".to_string()),
                }],
            ],
        })
    );
    assert_eq!(
        parse_command(
            "UPDATE public.people SET name = 'Updated', id = 10 WHERE id = 1 OR name LIKE 'Ada%'"
        )
        .unwrap(),
        Command::Update(Update {
            table: "people".to_string(),
            assignments: vec![
                UpdateAssignment {
                    column: "name".to_string(),
                    value: SqlValue::Text("Updated".to_string()),
                },
                UpdateAssignment {
                    column: "id".to_string(),
                    value: SqlValue::Int4(10),
                },
            ],
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }],
            filter_groups: vec![
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("Ada".to_string()),
                }],
            ],
        })
    );
    assert!(matches!(
        parse_command("UPDATE public.people SET name = 'Updated'"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("UPDATE public.people SET name = 'Updated' WHERE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert_eq!(
        parse_command("DELETE FROM balance").unwrap(),
        Command::DeleteKv {
            key: "balance".to_string(),
        }
    );
    assert_eq!(
        parse_command("DELETE FROM public.people").unwrap(),
        Command::DeleteKv {
            key: "public.people".to_string(),
        }
    );

    assert_eq!(
        parse_command("SELECT id FROM people WHERE id = +1 LIMIT +1").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }]],
            order_by: Vec::new(),
            limit: Some(1),
            offset: None,
        })
    );
    assert_eq!(
        parse_command("SELECT id FROM public.people WHERE id = 1").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }]],
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );
    assert_eq!(
        parse_command("SELECT id, name FROM ONLY public.people").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string(), "name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );
    assert!(matches!(
        parse_command("SELECT id FROM people ORDER BY id LIMIT -1"),
        Err(ParseError::NegativeLimit)
    ));
    assert!(matches!(
        parse_command("SELECT id FROM people ORDER BY id OFFSET -1"),
        Err(ParseError::NegativeOffset)
    ));

    assert_eq!(
        parse_command("SELECT id FROM people ORDER BY id LIMIT 2 OFFSET 1").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: Some(2),
            offset: Some(1),
        })
    );
    assert_eq!(
        parse_command("SELECT id FROM people ORDER BY id OFFSET 1 LIMIT 2").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: Some(2),
            offset: Some(1),
        })
    );

    assert_eq!(
        parse_command("SELECT id, name FROM people WHERE id = 1 ORDER BY name DESC LIMIT 5")
            .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string(), "name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }]],
            order_by: vec![SelectOrder {
                column: "name".to_string(),
                descending: true,
            }],
            limit: Some(5),
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE id >= 2 ORDER BY id LIMIT 5").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }]],
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: Some(5),
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE 2 <= id ORDER BY id LIMIT 5").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }]],
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: Some(5),
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE id >= 2 ORDER BY id LIMIT 5::pg_catalog.int4")
            .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }]],
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: Some(5),
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE name = 'O''Brien'").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Text("O'Brien".to_string()),
            }),
            filters: vec![SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Text("O'Brien".to_string()),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Text("O'Brien".to_string()),
            }]],
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert_eq!(
        parse_command(
            "SELECT id, name FROM people WHERE id = 2::int4 OR name = 'Ada'::text ORDER BY id"
        )
        .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string(), "name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(2),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(2),
            }],
            filter_groups: vec![
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(2),
                }],
                vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                }],
            ],
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: None,
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE id >= 2 AND name = 'Ada'").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![
                SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                },
                SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                },
            ],
            filter_groups: vec![vec![
                SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                },
                SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                },
            ]],
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE id = 1 OR name = 'Ada'").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }],
            filter_groups: vec![
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                }],
            ],
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE (id = 1) OR (name = 'Ada')").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }],
            filter_groups: vec![
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                }],
            ],
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert_eq!(
        parse_command(
            "SELECT id FROM people WHERE (id = 1 OR id = 3) AND (name = 'Ada' OR name = 'Grace') ORDER BY id"
        )
        .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![
                SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                },
                SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                },
            ],
            filter_groups: vec![
                vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(1),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                ],
                vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(1),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Grace".to_string()),
                    },
                ],
                vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(3),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                ],
                vec![
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Int4(3),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Grace".to_string()),
                    },
                ],
            ],
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: None,
            offset: None,
        })
    );
}
