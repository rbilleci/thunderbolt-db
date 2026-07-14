//! Bounded catalog SQL compatibility parser coverage.

use super::*;

#[test]
fn parses_bounded_sequence_catalog_ddl() {
    assert_eq!(
        parse_command("CREATE SEQUENCE public.seq_people").unwrap(),
        Command::CreateSequence(CreateSequence {
            name: "seq_people".to_string(),
        })
    );
    assert_eq!(
        parse_command(
            "CREATE SEQUENCE public.seq_people START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1"
        )
        .unwrap(),
        Command::CreateSequence(CreateSequence {
            name: "seq_people".to_string(),
        })
    );
    assert_eq!(
        parse_command("DROP SEQUENCE IF EXISTS public.seq_people, seq_teams").unwrap(),
        Command::DropSequence(DropSequence {
            names: vec!["seq_people".to_string(), "seq_teams".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("ALTER SEQUENCE public.seq_people RENAME TO seq_person_ids").unwrap(),
        Command::RenameSequence(RenameSequence {
            old_name: "seq_people".to_string(),
            new_name: "seq_person_ids".to_string(),
        })
    );
    assert_eq!(
        parse_command("COMMENT ON SEQUENCE public.seq_people IS 'ids'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Sequence {
                sequence: "seq_people".to_string(),
            },
            comment: Some("ids".to_string()),
        })
    );

    assert!(matches!(
        parse_command("CREATE SEQUENCE seq_people START WITH 10"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE SEQUENCE seq_people AS bigint"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP SEQUENCE seq_people CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER SEQUENCE IF EXISTS seq_people RENAME TO seq_person_ids"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER SEQUENCE seq_people RENAME TO public.seq_person_ids"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER SEQUENCE seq_people RENAME TO seq_person_ids CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_bounded_domain_catalog_ddl() {
    assert_eq!(
        parse_command("CREATE DOMAIN public.account_id AS int4").unwrap(),
        Command::CreateDomain(CreateDomain {
            name: "account_id".to_string(),
            base_type: SqlType::Int4,
        })
    );
    assert_eq!(
        parse_command("CREATE DOMAIN label AS pg_catalog.text").unwrap(),
        Command::CreateDomain(CreateDomain {
            name: "label".to_string(),
            base_type: SqlType::Text,
        })
    );
    assert_eq!(
        parse_command("COMMENT ON DOMAIN public.account_id IS 'domain ids'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Domain {
                domain: "account_id".to_string(),
            },
            comment: Some("domain ids".to_string()),
        })
    );
    assert_eq!(
        parse_command("DROP DOMAIN IF EXISTS public.account_id, label").unwrap(),
        Command::DropDomain(DropDomain {
            domains: vec!["account_id".to_string(), "label".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("CREATE TABLE accounts (id account_id, label public.label)").unwrap(),
        Command::CreateTable(CreateTable {
            table: "accounts".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    ty: SqlType::Int4,
                    domain: Some("account_id".to_string()),
                    default: None,
                },
                ColumnDef {
                    name: "label".to_string(),
                    ty: SqlType::Int4,
                    domain: Some("label".to_string()),
                    default: None,
                },
            ],
            primary_key: None,
            unique_constraints: Vec::new(),
            check_constraints: Vec::new(),
        })
    );

    assert!(matches!(
        parse_command("CREATE DOMAIN account_id AS int4 CHECK (VALUE > 0)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE TABLE accounts (id account_id DEFAULT 1)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    // `bigint` is a storable base type as of Phase-3 M1, so a domain over it now parses
    // (previously only int4/text were supported base types). A parenthesized typmod
    // (e.g. `numeric(12,2)`) on a domain stays rejected — domain typmods are out of M1 scope.
    assert_eq!(
        parse_command("CREATE DOMAIN account_id AS bigint").unwrap(),
        Command::CreateDomain(CreateDomain {
            name: "account_id".to_string(),
            base_type: SqlType::Int8,
        })
    );
    assert!(matches!(
        parse_command("CREATE DOMAIN money_amount AS numeric(12,2)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP DOMAIN account_id CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_bounded_function_catalog_ddl() {
    assert_eq!(
        parse_command("CREATE FUNCTION public.answer() RETURNS int4 LANGUAGE sql AS 'SELECT 42'")
            .unwrap(),
        Command::CreateFunction(CreateFunction {
            name: "answer".to_string(),
            return_type: SqlType::Int4,
            body: "SELECT 42".to_string(),
        })
    );
    assert_eq!(
        parse_command(
            "CREATE FUNCTION public.dump_answer() RETURNS integer LANGUAGE sql AS $$SELECT 42$$"
        )
        .unwrap(),
        Command::CreateFunction(CreateFunction {
            name: "dump_answer".to_string(),
            return_type: SqlType::Int4,
            body: "SELECT 42".to_string(),
        })
    );
    assert_eq!(
        parse_command("COMMENT ON FUNCTION public.answer() IS 'metadata only'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Function {
                function: "answer".to_string(),
            },
            comment: Some("metadata only".to_string()),
        })
    );
    assert_eq!(
        parse_command("COMMENT ON EXTENSION plpgsql IS 'bootstrap extension'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::Extension {
                extension: "plpgsql".to_string(),
            },
            comment: Some("bootstrap extension".to_string()),
        })
    );
    assert_eq!(
        parse_command("ALTER FUNCTION public.answer() RENAME TO ultimate_answer").unwrap(),
        Command::RenameFunction(RenameFunction {
            old_name: "answer".to_string(),
            new_name: "ultimate_answer".to_string(),
        })
    );
    assert_eq!(
        parse_command("DROP FUNCTION IF EXISTS public.answer()").unwrap(),
        Command::DropFunction(DropFunction {
            name: "answer".to_string(),
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("SELECT public.answer()").unwrap(),
        Command::SelectFunction(SelectFunction {
            name: "answer".to_string(),
        })
    );
    assert_eq!(
        parse_command("SELECT answer()").unwrap(),
        Command::SelectFunction(SelectFunction {
            name: "answer".to_string(),
        })
    );

    assert!(matches!(
        parse_command("CREATE FUNCTION public.echo(int4) RETURNS int4 LANGUAGE sql AS 'SELECT $1'"),
        Err(ParseError::InvalidRelationalSql)
    ));
    // `bigint` is a storable return type as of Phase-3 M1 (previously unsupported).
    assert_eq!(
        parse_command("CREATE FUNCTION public.answer() RETURNS bigint LANGUAGE sql AS 'SELECT 42'")
            .unwrap(),
        Command::CreateFunction(CreateFunction {
            name: "answer".to_string(),
            return_type: SqlType::Int8,
            body: "SELECT 42".to_string(),
        })
    );
    assert!(matches!(
        parse_command(
            "CREATE FUNCTION public.answer() RETURNS int4 LANGUAGE plpgsql AS 'BEGIN END'"
        ),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER FUNCTION public.answer(int4) RENAME TO answer2"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER FUNCTION public.answer() RENAME TO public.answer2"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER FUNCTION public.answer() OWNER TO postgres"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP FUNCTION public.answer() CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("SELECT answer(1)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("SELECT answer() FROM people"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_bounded_bootstrap_extension_create() {
    assert_eq!(
        parse_command("CREATE EXTENSION IF NOT EXISTS plpgsql").unwrap(),
        Command::CreateExtension(CreateExtension {
            name: "plpgsql".to_string(),
            if_not_exists: true,
            schema: None,
        })
    );
    assert_eq!(
        parse_command("CREATE EXTENSION IF NOT EXISTS \"plpgsql\" WITH SCHEMA pg_catalog").unwrap(),
        Command::CreateExtension(CreateExtension {
            name: "plpgsql".to_string(),
            if_not_exists: true,
            schema: Some("pg_catalog".to_string()),
        })
    );
    assert_eq!(
        parse_command("CREATE EXTENSION plpgsql").unwrap(),
        Command::CreateExtension(CreateExtension {
            name: "plpgsql".to_string(),
            if_not_exists: false,
            schema: None,
        })
    );

    assert!(matches!(
        parse_command("CREATE EXTENSION IF NOT EXISTS plpgsql VERSION '1.0'"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE EXTENSION IF NOT EXISTS plpgsql WITH VERSION '1.0'"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA public VERSION '1.0'"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_bounded_bootstrap_extension_drop_cleanup() {
    assert_eq!(
        parse_command("DROP EXTENSION IF EXISTS plpgsql").unwrap(),
        Command::DropExtension(DropExtension {
            name: "plpgsql".to_string(),
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("DROP EXTENSION \"plpgsql\"").unwrap(),
        Command::DropExtension(DropExtension {
            name: "plpgsql".to_string(),
            if_exists: false,
        })
    );

    assert!(matches!(
        parse_command("DROP EXTENSION IF EXISTS plpgsql CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP EXTENSION IF EXISTS plpgsql, hstore"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_bounded_sequence_value_functions() {
    assert_eq!(
        parse_command("SELECT nextval('public.seq_people'::regclass)").unwrap(),
        Command::SequenceNextVal(SequenceNextVal {
            name: "seq_people".to_string(),
        })
    );
    assert_eq!(
        parse_command("SELECT pg_catalog.currval('seq_people'::pg_catalog.regclass)").unwrap(),
        Command::SequenceCurrVal(SequenceCurrVal {
            name: "seq_people".to_string(),
        })
    );
    assert_eq!(
        parse_command("SELECT setval('public.seq_people', 42, false)").unwrap(),
        Command::SequenceSetVal(SequenceSetVal {
            name: "seq_people".to_string(),
            value: 42,
            is_called: false,
        })
    );
    assert_eq!(
        parse_command("SELECT pg_catalog.setval('public.seq_people', 42)").unwrap(),
        Command::SequenceSetVal(SequenceSetVal {
            name: "seq_people".to_string(),
            value: 42,
            is_called: true,
        })
    );

    assert!(matches!(
        parse_command("SELECT nextval('other.seq_people'::regclass)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("SELECT setval('seq_people', '42', false)"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("SELECT nextval('seq_people') FROM seq_people"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_bounded_materialized_view_lifecycle() {
    assert_eq!(
        parse_command(
            "CREATE MATERIALIZED VIEW public.mv_people AS SELECT id, name FROM people ORDER BY id"
        )
        .unwrap(),
        Command::CreateMaterializedView(CreateMaterializedView {
            name: "mv_people".to_string(),
            query: Select {
                table: "people".to_string(),
                distinct: false,
                projection: SelectProjection::Columns(vec!["id".to_string(), "name".to_string(),]),
                group_by: None,
                having_groups: Vec::new(),
                filter: None,
                filters: Vec::new(),
                filter_groups: Vec::new(),
                order_by: vec![SelectOrder {
                    column: "id".to_string(),
                    descending: false,
                }],
                limit: None,
                offset: None,
            },
            definition: "SELECT id, name FROM people ORDER BY id".to_string(),
            with_data: true,
        })
    );
    assert_eq!(
        parse_command("CREATE MATERIALIZED VIEW mv_people AS SELECT * FROM people WITH NO DATA")
            .unwrap(),
        Command::CreateMaterializedView(CreateMaterializedView {
            name: "mv_people".to_string(),
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
            with_data: false,
        })
    );
    assert_eq!(
        parse_command("ALTER MATERIALIZED VIEW public.mv_people RENAME TO mv_people_old").unwrap(),
        Command::RenameMaterializedView(RenameMaterializedView {
            old_name: "mv_people".to_string(),
            new_name: "mv_people_old".to_string(),
        })
    );
    assert_eq!(
        parse_command("REFRESH MATERIALIZED VIEW public.mv_people WITH DATA").unwrap(),
        Command::RefreshMaterializedView(RefreshMaterializedView {
            name: "mv_people".to_string(),
        })
    );
    assert_eq!(
        parse_command("DROP MATERIALIZED VIEW IF EXISTS public.mv_people_old, mv_other").unwrap(),
        Command::DropMaterializedView(DropMaterializedView {
            names: vec!["mv_people_old".to_string(), "mv_other".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_command("COMMENT ON MATERIALIZED VIEW public.mv_people IS 'snapshot'").unwrap(),
        Command::CommentOn(CommentOn {
            target: CommentTarget::MaterializedView {
                materialized_view: "mv_people".to_string(),
            },
            comment: Some("snapshot".to_string()),
        })
    );

    assert!(
        parse_command("CREATE MATERIALIZED VIEW mv_people AS SELECT * FROM people WITH DATA")
            .is_ok()
    );
    assert!(matches!(
        parse_command("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_people"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("REFRESH MATERIALIZED VIEW mv_people WITH NO DATA"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("DROP MATERIALIZED VIEW mv_people CASCADE"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("ALTER MATERIALIZED VIEW mv_people RENAME TO public.mv_people_old"),
        Err(ParseError::InvalidRelationalSql)
    ));
}
