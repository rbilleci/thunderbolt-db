//! SQL command terminator normalization tests.

use super::*;

#[test]
fn accepts_optional_statement_terminator() {
    assert_eq!(
        parse_command("BEGIN;").unwrap(),
        Command::Begin {
            characteristics: TransactionCharacteristics::default(),
        }
    );
    assert_eq!(
        parse_command("START WORK;").unwrap(),
        Command::Begin {
            characteristics: TransactionCharacteristics::default(),
        }
    );
    assert_eq!(
        parse_command("END AND CHAIN;").unwrap(),
        Command::Commit { chain: true }
    );
    assert_eq!(
        parse_command("ABORT AND NO CHAIN;\n").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("SET balance = 42;").unwrap(),
        Command::SetKv {
            key: "balance".into(),
            value: "42".into()
        }
    );
    assert_eq!(
        parse_command("GET balance;\n").unwrap(),
        Command::GetKv {
            key: "balance".into()
        }
    );
    assert_eq!(parse_command("RESET ALL;").unwrap(), Command::ResetAll);
    assert_eq!(parse_command("RESET ROLE;\n").unwrap(), Command::ResetAll);
    assert_eq!(
        parse_command("RESET SESSION ROLE;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("RESET LOCAL ROLE;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("RESET AUTHORIZATION;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(parse_command("RESET AUTH;\n").unwrap(), Command::ResetAll);
    assert_eq!(
        parse_command("RESET SESSION AUTHORIZATION;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("RESET SESSION AUTHORIZATION DEFAULT;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("RESET SESSION AUTHORIZATION TO DEFAULT;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("RESET SESSION AUTH;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("RESET SESSION AUTH DEFAULT;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("RESET SESSION AUTH TO DEFAULT;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(parse_command("DISCARD ALL;\n").unwrap(), Command::ResetAll);
    assert_eq!(parse_command("CLOSE ALL;\n").unwrap(), Command::ResetAll);
    assert_eq!(parse_command("UNLISTEN *;\n").unwrap(), Command::ResetAll);
    assert_eq!(parse_command("UNLISTEN ALL;\n").unwrap(), Command::ResetAll);
    assert_eq!(
        parse_command("LISTEN updates_channel;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("NOTIFY updates_channel;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("NOTIFY updates_channel, 'payload';\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("SET ROLE NONE;\n").unwrap(),
        Command::SetRole { role: None }
    );
    assert_eq!(
        parse_command("SET SESSION AUTHORIZATION DEFAULT;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("SET TRANSACTION READ ONLY;\n").unwrap(),
        Command::SessionControl {
            transaction: Some(TransactionCharacteristics {
                access: TransactionAccessMode::ReadOnly,
                ..TransactionCharacteristics::default()
            }),
            access_share_relations: Vec::new(),
        }
    );
    assert_eq!(
        parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;\n").unwrap(),
        Command::SessionControl {
            transaction: Some(TransactionCharacteristics {
                access: TransactionAccessMode::ReadOnly,
                ..TransactionCharacteristics::default()
            }),
            access_share_relations: Vec::new(),
        }
    );
    assert_eq!(
        parse_command("FLUSH WRITE AHEAD LOG;\n").unwrap(),
        Command::Flush
    );
    assert_eq!(parse_command("CHECKPOINT;\n").unwrap(), Command::Flush);
    assert_eq!(
        parse_command("FLUSH WRITE_AHEAD_LOG;\n").unwrap(),
        Command::Flush
    );
    assert_eq!(parse_command("DISCARD TEMP;\n").unwrap(), Command::ResetAll);
    assert_eq!(
        parse_command("DISCARD TEMP TABLES;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("DEALLOCATE ALL;\n").unwrap(),
        Command::ResetAll
    );
    assert_eq!(
        parse_command("DEALLOCATE PREPARE prepared_stmt;\n").unwrap(),
        Command::ResetAll
    );
}

#[test]
fn accepts_repeated_statement_terminators() {
    assert_eq!(
        parse_command("BEGIN;;").unwrap(),
        Command::Begin {
            characteristics: TransactionCharacteristics::default(),
        }
    );
    assert_eq!(
        parse_command("SET balance = 42; ; \n").unwrap(),
        Command::SetKv {
            key: "balance".into(),
            value: "42".into()
        }
    );
}

#[test]
fn rejects_input_that_is_only_terminators() {
    assert!(matches!(parse_command(";;;"), Err(ParseError::Empty)));
}
