//! SQL SET and session-control parser-facade tests.

use super::*;

fn transaction_control(characteristics: TransactionCharacteristics) -> Command {
    Command::SessionControl {
        transaction: Some(characteristics),
        access_share_relations: Vec::new(),
    }
}

#[test]
fn parses_set() {
    let cmd = parse_command("SET a = 42").unwrap();
    assert_eq!(
        cmd,
        Command::SetKv {
            key: "a".into(),
            value: "42".into()
        }
    );
}

#[test]
fn parses_set_with_non_space_whitespace_separator() {
    let cmd = parse_command("SET\ta = 42").unwrap();
    assert_eq!(
        cmd,
        Command::SetKv {
            key: "a".into(),
            value: "42".into()
        }
    );
}

#[test]
fn parses_set_with_to_assignment_alias() {
    let cmd = parse_command("SET a TO 42").unwrap();
    assert_eq!(
        cmd,
        Command::SetKv {
            key: "a".into(),
            value: "42".into()
        }
    );

    let cmd = parse_command("SET alpha to value words").unwrap();
    assert_eq!(
        cmd,
        Command::SetKv {
            key: "alpha".into(),
            value: "value words".into()
        }
    );
}

#[test]
fn parses_set_with_session_or_local_scope_aliases() {
    let cmd = parse_command("SET LOCAL a = 42").unwrap();
    assert_eq!(
        cmd,
        Command::SetKv {
            key: "a".into(),
            value: "42".into()
        }
    );

    let cmd = parse_command("SET SESSION a TO 42").unwrap();
    assert_eq!(
        cmd,
        Command::SetKv {
            key: "a".into(),
            value: "42".into()
        }
    );

    let cmd = parse_command("SET SESSION statement_timeout = 5s").unwrap();
    assert_eq!(
        cmd,
        Command::SetKv {
            key: "statement_timeout".into(),
            value: "5s".into()
        }
    );
}

#[test]
fn parses_postgres_style_set_session_reset_aliases() {
    assert_eq!(
        parse_command("SET ROLE NONE").unwrap(),
        Command::SetRole {
            role: None,
            scope: SetRoleScope::Session
        }
    );
    assert_eq!(
        parse_command("SET ROLE DEFAULT").unwrap(),
        Command::SetRole {
            role: None,
            scope: SetRoleScope::Session
        }
    );
    assert_eq!(
        parse_command("SET ROLE app_role").unwrap(),
        Command::SetRole {
            role: Some("app_role".to_string()),
            scope: SetRoleScope::Session,
        }
    );
    assert_eq!(
        parse_command("SET ROLE \"app role\"").unwrap(),
        Command::SetRole {
            role: Some("app role".to_string()),
            scope: SetRoleScope::Session,
        }
    );
    assert_eq!(
        parse_command("SET ROLE \"\"\"quoted\"\" role\"").unwrap(),
        Command::SetRole {
            role: Some("\"quoted\" role".to_string()),
            scope: SetRoleScope::Session,
        }
    );
    assert_eq!(
        parse_command("SET SESSION ROLE DEFAULT").unwrap(),
        Command::SetRole {
            role: None,
            scope: SetRoleScope::Session
        }
    );
    assert_eq!(
        parse_command("SET SESSION ROLE \"app role\"").unwrap(),
        Command::SetRole {
            role: Some("app role".to_string()),
            scope: SetRoleScope::Session,
        }
    );
    assert_eq!(
        parse_command("SET LOCAL ROLE NONE").unwrap(),
        Command::SetRole {
            role: None,
            scope: SetRoleScope::Local
        }
    );
    assert_eq!(
        parse_command("SET LOCAL ROLE DEFAULT").unwrap(),
        Command::SetRole {
            role: None,
            scope: SetRoleScope::Local
        }
    );
    assert_eq!(
        parse_command("SET LOCAL ROLE app_role").unwrap(),
        Command::SetRole {
            role: Some("app_role".to_string()),
            scope: SetRoleScope::Local,
        }
    );
    assert_eq!(
        parse_command("SET SESSION AUTHORIZATION DEFAULT").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("SET SESSION AUTH postgres").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("SET SESSION AUTHORIZATION \"app user\"").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("SET SESSION AUTH \"app user\"").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED")
            .unwrap(),
        transaction_control(TransactionCharacteristics::default())
    );
    assert_eq!(
        parse_command("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ").unwrap(),
        transaction_control(TransactionCharacteristics {
            isolation: TransactionIsolation::RepeatableRead,
            ..TransactionCharacteristics::default()
        })
    );
    assert_eq!(
        parse_command("SET TRANSACTION READ ONLY, DEFERRABLE").unwrap(),
        transaction_control(TransactionCharacteristics {
            access: TransactionAccessMode::ReadOnly,
            deferrable: true,
            ..TransactionCharacteristics::default()
        })
    );
    assert_eq!(
        parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE, NOT DEFERRABLE")
            .unwrap(),
        transaction_control(TransactionCharacteristics::default())
    );
    assert_eq!(
        parse_command("SET LOCAL TRANSACTION READ ONLY").unwrap(),
        transaction_control(TransactionCharacteristics {
            access: TransactionAccessMode::ReadOnly,
            ..TransactionCharacteristics::default()
        })
    );
    assert_eq!(
        parse_command("SET LOCAL TRANSACTION READ WRITE, DEFERRABLE").unwrap(),
        transaction_control(TransactionCharacteristics {
            deferrable: true,
            ..TransactionCharacteristics::default()
        })
    );
}

#[test]
fn rejects_set_with_whitespace_in_key() {
    assert!(matches!(
        parse_command("SET two words=42"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET a TO"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET a TO42"),
        Err(ParseError::InvalidSet)
    ));
}

#[test]
fn parses_flush() {
    let cmd = parse_command("CHECKPOINT").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WAL").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH LOG").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE AHEAD").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE AHEAD LOG").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE AHEAD WAL").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE-AHEAD").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE-AHEAD LOG").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE-AHEAD WAL").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITEAHEAD").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITEAHEAD LOG").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITEAHEAD WAL").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE_AHEAD").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE_AHEAD LOG").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE_AHEAD WAL").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE_AHEAD_LOG").unwrap();
    assert_eq!(cmd, Command::Flush);

    let cmd = parse_command("FLUSH WRITE_AHEAD_WAL").unwrap();
    assert_eq!(cmd, Command::Flush);
}

#[test]
fn parses_reset_all() {
    let session_role_reset = Command::SetRole {
        role: None,
        scope: SetRoleScope::Session,
    };
    let local_role_reset = Command::SetRole {
        role: None,
        scope: SetRoleScope::Local,
    };

    let cmd = parse_command("RESET ALL").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DISCARD ALL").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("RESET ROLE").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET SESSION ROLE").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET LOCAL ROLE").unwrap();
    assert_eq!(cmd, local_role_reset);

    let cmd = parse_command("RESET AUTHORIZATION").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET AUTH").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET SESSION AUTHORIZATION").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET SESSION AUTHORIZATION DEFAULT").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET SESSION AUTHORIZATION TO DEFAULT").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET SESSION AUTH").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET SESSION AUTH DEFAULT").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("RESET SESSION AUTH TO DEFAULT").unwrap();
    assert_eq!(cmd, session_role_reset);

    let cmd = parse_command("DISCARD TEMP").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DISCARD TEMPORARY").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DISCARD TEMP TABLE").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DISCARD TEMP TABLES").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DISCARD TEMPORARY TABLE").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DISCARD TEMPORARY TABLES").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DISCARD PLANS").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DISCARD SEQUENCES").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DEALLOCATE ALL").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DEALLOCATE prepared_stmt").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DEALLOCATE PREPARE prepared_stmt").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DEALLOCATE PREPARED prepared_stmt").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DEALLOCATE \"prepared stmt\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DEALLOCATE PREPARE \"prepared stmt\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("DEALLOCATE PREPARED \"prepared stmt\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("CLOSE ALL").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("CLOSE cursor_name").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("CLOSE \"cursor name\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("CLOSE \"cursor\"\"name\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("CLOSE \"\"\"quoted\"\" cursor\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("UNLISTEN").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("UNLISTEN *").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("UNLISTEN ALL").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("UNLISTEN updates_channel").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("UNLISTEN \"updates channel\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("UNLISTEN \"updates,channel\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("UNLISTEN \"updates\"\"channel\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("UNLISTEN \"updates\nΔetail\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("LISTEN updates_channel").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("LISTEN \"updates channel\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("LISTEN \"updates\"\"channel\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("LISTEN \"updates\nΔetail\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY \"updates channel\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY \"updates\"\"channel\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY \"updates\nΔetail\", 'héllo\nΔetail'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY \"updates,channel\", 'hello'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, 'hello'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel , 'hello'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel,'hello'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel ,'{\"ok\":true}'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, '{\"ok\":true,\"n\":1}'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, $$hello,world$$").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, $tag$hello,world$tag$").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, \"hello,world\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, \"hello\"\"world\"").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, E'hello\\'world'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, B'101010'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, X'CAFE'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, U&'d\\0061ta'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, e'hello\\'world'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, b'101010'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, x'cafe'").unwrap();
    assert_eq!(cmd, Command::ResetAll);

    let cmd = parse_command("NOTIFY updates_channel, u&'d\\0061ta'").unwrap();
    assert_eq!(cmd, Command::ResetAll);
}
