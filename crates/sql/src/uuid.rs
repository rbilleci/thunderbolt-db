//! Minimal `uuid` parse/format for the type matrix (doc 19). A `uuid` is the 16 raw bytes of the
//! 128-bit value; PostgreSQL compares two uuids by those bytes in order (an unsigned big-endian
//! `memcmp`), which is the natural `Ord` of `[u8; 16]`. We hand-roll the hex parsing (no `uuid` crate).

/// Parse a `uuid` literal to its 16 raw bytes. Accepts the canonical hyphenated form
/// (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`), the bare 32-hex form, and an optional `{...}` wrapper;
/// hyphens anywhere are ignored. Requires EXACTLY 32 hex digits. `None` otherwise -- the caller raises
/// PG's "invalid input syntax for type uuid".
pub fn parse_uuid(text: &str) -> Option<[u8; 16]> {
    let trimmed = text.trim();
    let body = trimmed
        .strip_prefix('{')
        .and_then(|inner| inner.strip_suffix('}'))
        .unwrap_or(trimmed);
    let mut bytes = [0u8; 16];
    let mut idx = 0usize;
    let mut high: Option<u8> = None;
    for ch in body.chars() {
        if ch == '-' {
            continue;
        }
        let nibble = u8::try_from(ch.to_digit(16)?).ok()?;
        match high.take() {
            None => high = Some(nibble),
            Some(hi) => {
                if idx >= 16 {
                    return None; // more than 32 hex digits
                }
                bytes[idx] = (hi << 4) | nibble;
                idx += 1;
            }
        }
    }
    if idx == 16 && high.is_none() {
        Some(bytes)
    } else {
        None // too few hex digits, or an odd count
    }
}

/// Format 16 raw bytes as the canonical lowercase hyphenated `uuid` text PostgreSQL emits.
pub fn format_uuid(bytes: &[u8; 16]) -> String {
    let mut out = String::with_capacity(36);
    for (i, byte) in bytes.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            out.push('-');
        }
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_round_trips_canonical() {
        let s = "550e8400-e29b-41d4-a716-446655440000";
        let bytes = parse_uuid(s).expect("valid uuid");
        assert_eq!(
            bytes,
            [
                0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
                0x00, 0x00
            ]
        );
        assert_eq!(format_uuid(&bytes), s, "round-trip canonical lowercase");
    }

    #[test]
    fn uuid_accepts_variants() {
        let canonical = parse_uuid("550e8400-e29b-41d4-a716-446655440000").unwrap();
        // Bare 32-hex, braces, and uppercase all parse to the same bytes.
        assert_eq!(parse_uuid("550e8400e29b41d4a716446655440000"), Some(canonical));
        assert_eq!(
            parse_uuid("{550e8400-e29b-41d4-a716-446655440000}"),
            Some(canonical)
        );
        assert_eq!(
            parse_uuid("550E8400-E29B-41D4-A716-446655440000"),
            Some(canonical),
            "uppercase hex"
        );
        // format is always lowercase canonical.
        assert_eq!(
            format_uuid(&canonical),
            "550e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[test]
    fn uuid_ordering_is_bytewise() {
        let lo = parse_uuid("00000000-0000-0000-0000-000000000001").unwrap();
        let hi = parse_uuid("ffffffff-ffff-ffff-ffff-ffffffffffff").unwrap();
        let mid = parse_uuid("80000000-0000-0000-0000-000000000000").unwrap();
        assert!(lo < mid && mid < hi, "byte-wise unsigned ordering");
        assert_eq!(parse_uuid("00000000-0000-0000-0000-000000000000").unwrap(), [0u8; 16]);
    }

    #[test]
    fn uuid_rejects_malformed() {
        assert_eq!(parse_uuid(""), None);
        assert_eq!(parse_uuid("550e8400"), None, "too few digits");
        assert_eq!(
            parse_uuid("550e8400-e29b-41d4-a716-4466554400000"),
            None,
            "33 hex digits"
        );
        assert_eq!(
            parse_uuid("550e8400-e29b-41d4-a716-44665544000g"),
            None,
            "non-hex digit"
        );
        assert_eq!(
            parse_uuid("550e8400-e29b-41d4-a716-44665544000"),
            None,
            "31 hex digits"
        );
    }
}
