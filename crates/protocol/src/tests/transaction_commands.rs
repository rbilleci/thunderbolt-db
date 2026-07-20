//! Transaction-control parser-facade tests.

use super::*;

fn begin(characteristics: TransactionCharacteristics) -> Command {
    Command::Begin { characteristics }
}

#[test]
fn parses_transaction_control_commands_case_insensitively() {
    assert_eq!(
        parse_command("begin").unwrap(),
        begin(TransactionCharacteristics::default())
    );
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
    let default = TransactionCharacteristics::default();
    let read_only = TransactionCharacteristics {
        access: TransactionAccessMode::ReadOnly,
        ..default
    };
    let serializable = TransactionCharacteristics {
        isolation: TransactionIsolation::Serializable,
        ..default
    };
    let repeatable_read = TransactionCharacteristics::REPEATABLE_READ_WRITE;
    let read_uncommitted = TransactionCharacteristics {
        isolation: TransactionIsolation::ReadUncommitted,
        ..default
    };
    let deferrable = TransactionCharacteristics {
        deferrable: true,
        ..default
    };
    for (sql, expected) in [
        ("BEGIN WORK", default),
        ("BEGIN TRANSACTION", default),
        ("BEGIN READ ONLY", read_only),
        ("BEGIN READ WRITE", default),
        ("BEGIN ISOLATION LEVEL SERIALIZABLE", serializable),
        ("BEGIN ISOLATION LEVEL REPEATABLE READ", repeatable_read),
        ("BEGIN ISOLATION LEVEL READ COMMITTED", default),
        ("BEGIN ISOLATION LEVEL READ UNCOMMITTED", read_uncommitted),
        ("BEGIN DEFERRABLE", deferrable),
        ("BEGIN NOT DEFERRABLE", default),
        ("BEGIN TRANSACTION READ ONLY", read_only),
        ("BEGIN WORK READ WRITE", default),
        (
            "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE",
            serializable,
        ),
        (
            "BEGIN READ WRITE, ISOLATION LEVEL SERIALIZABLE",
            serializable,
        ),
        (
            "BEGIN READ ONLY , DEFERRABLE",
            TransactionCharacteristics {
                access: TransactionAccessMode::ReadOnly,
                deferrable: true,
                ..default
            },
        ),
        (
            "BEGIN READ ONLY DEFERRABLE",
            TransactionCharacteristics {
                access: TransactionAccessMode::ReadOnly,
                deferrable: true,
                ..default
            },
        ),
        (
            "BEGIN READ WRITE ISOLATION LEVEL SERIALIZABLE",
            serializable,
        ),
        ("START TRANSACTION", default),
        ("BEGIN TRANSACTION, READ ONLY", read_only),
        ("BEGIN WORK, READ WRITE", default),
        ("START TRANSACTION READ ONLY", read_only),
        ("START TRANSACTION, READ ONLY", read_only),
        (
            "START TRANSACTION ISOLATION LEVEL REPEATABLE READ",
            repeatable_read,
        ),
        ("START WORK", default),
        ("START WORK, READ WRITE, DEFERRABLE", deferrable),
        ("START WORK ISOLATION LEVEL READ COMMITTED", default),
        (
            "START WORK, ISOLATION LEVEL REPEATABLE READ, NOT DEFERRABLE",
            repeatable_read,
        ),
        ("START TRANSACTION DEFERRABLE", deferrable),
        (
            "START TRANSACTION READ ONLY DEFERRABLE",
            TransactionCharacteristics {
                access: TransactionAccessMode::ReadOnly,
                deferrable: true,
                ..default
            },
        ),
        ("START WORK READ WRITE", default),
    ] {
        assert_eq!(parse_command(sql).unwrap(), begin(expected), "{sql}");
    }
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
    let default = TransactionCharacteristics::default();
    assert_eq!(
        parse_command("BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE, READ ONLY, NOT DEFERRABLE")
            .unwrap(),
        begin(TransactionCharacteristics {
            isolation: TransactionIsolation::Serializable,
            access: TransactionAccessMode::ReadOnly,
            deferrable: false,
        })
    );
    assert_eq!(
        parse_command("BEGIN READ WRITE ISOLATION LEVEL READ COMMITTED DEFERRABLE").unwrap(),
        begin(TransactionCharacteristics {
            deferrable: true,
            ..default
        })
    );
    assert_eq!(
        parse_command("START WORK, NOT DEFERRABLE, ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            .unwrap(),
        begin(TransactionCharacteristics {
            isolation: TransactionIsolation::RepeatableRead,
            access: TransactionAccessMode::ReadOnly,
            deferrable: false,
        })
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
