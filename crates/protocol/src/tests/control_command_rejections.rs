//! Negative parser matrix for SQL control and session commands.

use super::*;

#[test]
fn rejects_transaction_control_commands_with_extra_tokens() {
    assert!(matches!(
        parse_command("BEGIN TRANSACTION NOW"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN READ"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN READ COMMITTED"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN ISOLATION LEVEL"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN ISOLATION LEVEL SNAPSHOT"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN NOT"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN READ ONLY, "),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN , READ ONLY"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN READ ONLY,, DEFERRABLE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("START TRANSACTION READ"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN TRANSACTION,"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("START WORK, "),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("START TRANSACTION READ COMMITTED"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("START WORK NOW"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN READ ONLY READ WRITE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN READ ONLY, READ WRITE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("START TRANSACTION READ ONLY, READ ONLY"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("START WORK ISOLATION LEVEL READ COMMITTED, ISOLATION LEVEL SERIALIZABLE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN DEFERRABLE NOT DEFERRABLE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("BEGIN ISOLATION LEVEL SERIALIZABLE, ISOLATION LEVEL READ COMMITTED"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("COMMIT WORK PLEASE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("COMMIT AND"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("COMMIT AND MAYBE CHAIN"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("COMMIT TRANSACTION AND"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("END WORK PLEASE"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("END AND"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("END WORK AND"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("ROLLBACK TRANSACTION AGAIN"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("ROLLBACK AND"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("ROLLBACK WORK AND"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("ABORT TRANSACTION AGAIN"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("ABORT AND"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("ABORT WORK AND"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("FLUSH NOW"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("FLUSH WAL NOW"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("CHECKPOINT NOW"),
        Err(ParseError::Unsupported(_))
    ));
    assert!(matches!(
        parse_command("RESET"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("RESET SESSION"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("RESET SESSION AUTHORIZATION DEFAULT NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("RESET SESSION AUTHORIZATION TO DEFAULT NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("RESET SESSION AUTH DEFAULT NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("RESET SESSION AUTH TO DEFAULT NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("RESET ALL NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DISCARD"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DISCARD TEMP NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DISCARD ALL NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE PREPARE"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE PREPARE x y"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE PREPARED"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE PREPARED x y"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE a,b"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE PREPARE a,b"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE \"prepared stmt"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("DEALLOCATE \"\""),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("CLOSE"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("CLOSE cursor_name NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("CLOSE \"cursor name"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("CLOSE \"\""),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("UNLISTEN * NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("UNLISTEN a,b"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("UNLISTEN \"updates channel"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("UNLISTEN \"\""),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("LISTEN"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("LISTEN updates_channel NOW"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("LISTEN \"\""),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel payload"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel ,"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel,"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, ,"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel , ,"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, , ,"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel,,payload"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel,, payload"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, payload, extra"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, payload extra"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, 'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, \"unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, $$unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, $tag$unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, E'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, B'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, X'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, U&'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, e'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, b'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, x'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY updates_channel, u&'unterminated"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY ;"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("LISTEN ;"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("UNLISTEN @invalid"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY \"updates channel"),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("NOTIFY \"\""),
        Err(ParseError::InvalidReset)
    ));
    assert!(matches!(
        parse_command("SET ROLE"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET SESSION ROLE"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET LOCAL ROLE"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET SESSION AUTHORIZATION"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET SESSION AUTH"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET ROLE \"unterminated"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET ROLE \"\""),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET SESSION AUTHORIZATION \"unterminated"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET TRANSACTION"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET TRANSACTION NOW"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET TRANSACTION READ ONLY, READ WRITE"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET LOCAL TRANSACTION"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET LOCAL TRANSACTION NOW"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION"),
        Err(ParseError::InvalidSet)
    ));
    assert!(matches!(
        parse_command("SET SESSION CHARACTERISTICS AS TRANSACTION NOW"),
        Err(ParseError::InvalidSet)
    ));
}
