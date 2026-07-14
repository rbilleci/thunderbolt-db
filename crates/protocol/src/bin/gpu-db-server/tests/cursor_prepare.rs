use super::*;

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
