//! Legacy key-value parser-facade command tests.

use super::*;

#[test]
fn parses_del() {
    let cmd = parse_command("DEL balance").unwrap();
    assert_eq!(
        cmd,
        Command::DeleteKv {
            key: "balance".into()
        }
    );
}

#[test]
fn parses_delete_alias() {
    let cmd = parse_command("DELETE balance").unwrap();
    assert_eq!(
        cmd,
        Command::DeleteKv {
            key: "balance".into()
        }
    );
}

#[test]
fn parses_delete_from_alias() {
    let cmd = parse_command("DELETE FROM balance").unwrap();
    assert_eq!(
        cmd,
        Command::DeleteKv {
            key: "balance".into()
        }
    );
}

#[test]
fn rejects_del_with_missing_or_extra_tokens() {
    assert!(matches!(parse_command("DEL"), Err(ParseError::InvalidDel)));
    assert!(matches!(
        parse_command("DEL too many"),
        Err(ParseError::InvalidDel)
    ));
    assert!(matches!(
        parse_command("DELETE FROM"),
        Err(ParseError::InvalidDel)
    ));
    assert!(matches!(
        parse_command("DELETE FROM too many"),
        Err(ParseError::InvalidDel)
    ));
    assert!(matches!(
        parse_command("DELETE TABLE balance"),
        Err(ParseError::InvalidDel)
    ));
}

#[test]
fn parses_get() {
    let cmd = parse_command("GET balance").unwrap();
    assert_eq!(
        cmd,
        Command::GetKv {
            key: "balance".into()
        }
    );
}

#[test]
fn rejects_get_with_missing_or_extra_tokens() {
    assert!(matches!(parse_command("GET"), Err(ParseError::InvalidGet)));
    assert!(matches!(
        parse_command("GET too many"),
        Err(ParseError::InvalidGet)
    ));
}
