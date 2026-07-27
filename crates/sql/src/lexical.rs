//! Allocation-free lexical gates for compatibility classifiers.
//!
//! These gates answer only whether a statement *may* begin with one of a small set of keywords.
//! A positive result deliberately leaves syntax ownership with the established classifier. An
//! unterminated or depth-overflowed leading block comment is indeterminate and therefore returns
//! `true`, preserving the old slow-path error behavior.

/// Return whether the first unquoted identifier after leading whitespace and comments may match
/// one of `candidates`, case-insensitively.
///
/// Leading `--` comments terminate on either CR or LF. Block comments may nest. A malformed
/// leading block comment is deliberately fail-open so callers retain their existing parser path.
/// Candidate prefixes followed by a non-ASCII character are also fail-open: legacy compatibility
/// classifiers do not agree whether every non-ASCII character continues an identifier.
pub fn sql_may_start_with_any_keyword(sql: &str, candidates: &[&str]) -> bool {
    let mut offset = 0;
    loop {
        offset = skip_unicode_whitespace(sql, offset);
        let rest = &sql[offset..];
        if rest.starts_with("--") {
            offset = skip_line_comment(sql, offset);
            continue;
        }
        if rest.starts_with("/*") {
            let Some(after_comment) = skip_nested_block_comment(sql, offset) else {
                return true;
            };
            offset = after_comment;
            continue;
        }
        break;
    }

    let token_start = offset;
    let Some(first) = sql[token_start..].chars().next() else {
        return false;
    };
    if !is_identifier_start(first) {
        return false;
    }
    offset += first.len_utf8();
    while let Some(next) = sql[offset..].chars().next() {
        if !is_identifier_continue(next) {
            break;
        }
        offset += next.len_utf8();
    }
    let token = &sql[token_start..offset];
    candidates.iter().any(|candidate| {
        token.eq_ignore_ascii_case(candidate)
            || token
                .get(..candidate.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(candidate))
                && token[candidate.len()..]
                    .chars()
                    .next()
                    .is_some_and(|character| !character.is_ascii())
    })
}

fn skip_unicode_whitespace(sql: &str, mut offset: usize) -> usize {
    while let Some(character) = sql[offset..].chars().next() {
        if !character.is_whitespace() {
            break;
        }
        offset += character.len_utf8();
    }
    offset
}

fn skip_line_comment(sql: &str, offset: usize) -> usize {
    let bytes = sql.as_bytes();
    let mut cursor = offset + 2;
    while cursor < bytes.len() && bytes[cursor] != b'\r' && bytes[cursor] != b'\n' {
        cursor += 1;
    }
    cursor
}

/// `None` is the intentional indeterminate/fail-open result.
fn skip_nested_block_comment(sql: &str, offset: usize) -> Option<usize> {
    let bytes = sql.as_bytes();
    let mut cursor = offset.checked_add(2)?;
    let mut depth = 1_usize;
    while cursor < bytes.len() {
        if bytes[cursor..].starts_with(b"/*") {
            depth = depth.checked_add(1)?;
            cursor = cursor.checked_add(2)?;
        } else if bytes[cursor..].starts_with(b"*/") {
            depth -= 1;
            cursor = cursor.checked_add(2)?;
            if depth == 0 {
                return Some(cursor);
            }
        } else {
            cursor = cursor.checked_add(1)?;
        }
    }
    None
}

fn is_identifier_start(character: char) -> bool {
    character.is_ascii_alphabetic() || character == '_' || !character.is_ascii()
}

fn is_identifier_continue(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(character, '_' | '$')
        || (!character.is_ascii() && !character.is_whitespace())
}

#[cfg(test)]
mod tests {
    use super::sql_may_start_with_any_keyword;

    const COMPAT: &[&str] = &["COPY", "DECLARE", "PREPARE", "EXECUTE"];

    #[test]
    fn accepts_complete_keywords_after_unicode_whitespace_and_nested_comments() {
        assert!(sql_may_start_with_any_keyword(
            "\u{2003}\tCoPy\u{00a0}\u{202f}table",
            COMPAT
        ));
        assert!(sql_may_start_with_any_keyword(
            "-- lead\r/* outer /* inner */ */\nPREPARE\u{2003}\u{1680}item AS SELECT 1",
            COMPAT,
        ));
    }

    #[test]
    fn rejects_identifier_extensions_and_non_identifier_starts() {
        for sql in [
            "COPYfoo table",
            "COPY_ table",
            "COPY$1 table",
            "\"COPY\" table",
            "'COPY'",
            "$tag$COPY$tag$",
            "(COPY table)",
            "-- comment only\r",
        ] {
            assert!(
                !sql_may_start_with_any_keyword(sql, COMPAT),
                "{sql:?} must not enter a keyword classifier"
            );
        }
    }

    #[test]
    fn non_ascii_candidate_extensions_fail_open_for_compatibility_classifiers() {
        assert!(sql_may_start_with_any_keyword("COPYé table", COMPAT));
        assert!(sql_may_start_with_any_keyword("DECLAREλ cursor", COMPAT));
    }

    #[test]
    fn unterminated_nested_block_comment_fails_open() {
        assert!(sql_may_start_with_any_keyword(
            "/* one /* two */ still open",
            COMPAT,
        ));
    }
}
