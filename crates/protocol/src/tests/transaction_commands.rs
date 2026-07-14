//! Transaction-control parser-facade tests.

use super::*;

#[test]
fn parses_transaction_control_commands_case_insensitively() {
    assert_eq!(parse_command("begin").unwrap(), Command::Begin);
    assert_eq!(
        parse_command("COMMIT").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("END").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("rOlLbAcK").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("abort").unwrap(),
        Command::Rollback { chain: false }
    );
}

#[test]
fn parses_transaction_control_work_and_transaction_aliases() {
    assert_eq!(parse_command("BEGIN WORK").unwrap(), Command::Begin);
    assert_eq!(parse_command("BEGIN TRANSACTION").unwrap(), Command::Begin);
    assert_eq!(parse_command("BEGIN READ ONLY").unwrap(), Command::Begin);
    assert_eq!(parse_command("BEGIN READ WRITE").unwrap(), Command::Begin);
    assert_eq!(
        parse_command("BEGIN ISOLATION LEVEL SERIALIZABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN ISOLATION LEVEL REPEATABLE READ").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN ISOLATION LEVEL READ COMMITTED").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN ISOLATION LEVEL READ UNCOMMITTED").unwrap(),
        Command::Begin
    );
    assert_eq!(parse_command("BEGIN DEFERRABLE").unwrap(), Command::Begin);
    assert_eq!(
        parse_command("BEGIN NOT DEFERRABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN TRANSACTION READ ONLY").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN WORK READ WRITE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN READ WRITE, ISOLATION LEVEL SERIALIZABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN READ ONLY , DEFERRABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN READ ONLY DEFERRABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN READ WRITE ISOLATION LEVEL SERIALIZABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(parse_command("START TRANSACTION").unwrap(), Command::Begin);
    assert_eq!(
        parse_command("BEGIN TRANSACTION, READ ONLY").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN WORK, READ WRITE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START TRANSACTION READ ONLY").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START TRANSACTION, READ ONLY").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START TRANSACTION ISOLATION LEVEL REPEATABLE READ").unwrap(),
        Command::Begin
    );
    assert_eq!(parse_command("START WORK").unwrap(), Command::Begin);
    assert_eq!(
        parse_command("START WORK, READ WRITE, DEFERRABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START WORK ISOLATION LEVEL READ COMMITTED").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START WORK, ISOLATION LEVEL REPEATABLE READ, NOT DEFERRABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START TRANSACTION DEFERRABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START TRANSACTION READ ONLY DEFERRABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START WORK READ WRITE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("COMMIT WORK").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("COMMIT TRANSACTION").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("COMMIT AND CHAIN").unwrap(),
        Command::Commit { chain: true }
    );
    assert_eq!(
        parse_command("COMMIT AND NO CHAIN").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("COMMIT TRANSACTION AND CHAIN").unwrap(),
        Command::Commit { chain: true }
    );
    assert_eq!(
        parse_command("COMMIT WORK AND CHAIN").unwrap(),
        Command::Commit { chain: true }
    );
    assert_eq!(
        parse_command("COMMIT WORK AND NO CHAIN").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("END WORK").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("END TRANSACTION").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("END AND CHAIN").unwrap(),
        Command::Commit { chain: true }
    );
    assert_eq!(
        parse_command("END AND NO CHAIN").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("END TRANSACTION AND CHAIN").unwrap(),
        Command::Commit { chain: true }
    );
    assert_eq!(
        parse_command("END WORK AND CHAIN").unwrap(),
        Command::Commit { chain: true }
    );
    assert_eq!(
        parse_command("END WORK AND NO CHAIN").unwrap(),
        Command::Commit { chain: false }
    );
    assert_eq!(
        parse_command("ROLLBACK WORK").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("ROLLBACK TRANSACTION").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("ROLLBACK AND CHAIN").unwrap(),
        Command::Rollback { chain: true }
    );
    assert_eq!(
        parse_command("ROLLBACK AND NO CHAIN").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("ROLLBACK TRANSACTION AND CHAIN").unwrap(),
        Command::Rollback { chain: true }
    );
    assert_eq!(
        parse_command("ROLLBACK WORK AND CHAIN").unwrap(),
        Command::Rollback { chain: true }
    );
    assert_eq!(
        parse_command("ROLLBACK WORK AND NO CHAIN").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("ABORT WORK").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("ABORT TRANSACTION").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("ABORT AND CHAIN").unwrap(),
        Command::Rollback { chain: true }
    );
    assert_eq!(
        parse_command("ABORT AND NO CHAIN").unwrap(),
        Command::Rollback { chain: false }
    );
    assert_eq!(
        parse_command("ABORT TRANSACTION AND CHAIN").unwrap(),
        Command::Rollback { chain: true }
    );
    assert_eq!(
        parse_command("ABORT WORK AND CHAIN").unwrap(),
        Command::Rollback { chain: true }
    );
    assert_eq!(
        parse_command("ABORT WORK AND NO CHAIN").unwrap(),
        Command::Rollback { chain: false }
    );
}

#[test]
fn parses_begin_mode_lists_with_mixed_order_and_delimiters() {
    assert_eq!(
        parse_command("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ ONLY, NOT DEFERRABLE")
            .unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("BEGIN READ WRITE ISOLATION LEVEL READ COMMITTED DEFERRABLE").unwrap(),
        Command::Begin
    );
    assert_eq!(
        parse_command("START WORK, NOT DEFERRABLE, ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            .unwrap(),
        Command::Begin
    );
}

#[test]
fn rejects_begin_mode_lists_with_duplicate_mode_kinds_even_when_comma_delimited() {
    assert!(matches!(
        parse_command("BEGIN READ ONLY, READ WRITE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN DEFERRABLE, NOT DEFERRABLE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command(
            "START TRANSACTION ISOLATION LEVEL READ COMMITTED, ISOLATION LEVEL SERIALIZABLE"
        ),
        Err(ParseError::Unsupported(_))
    ));
}
