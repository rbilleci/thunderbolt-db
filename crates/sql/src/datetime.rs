//! Minimal proleptic-Gregorian calendar math for the temporal types (the type matrix, doc 19).
//! Hand-rolled (no `chrono`/`time` dependency): the conversions are Howard Hinnant's standard
//! `days_from_civil` / `civil_from_days` algorithms. A `DATE` is stored as i32 DAYS since 2000-01-01
//! (the PostgreSQL date epoch), so byte-for-byte it is an i32 column and reuses the int4 device path;
//! a `TIMESTAMP` is i64 MICROSECONDS since 2000-01-01 00:00:00, reusing the int8 device path.

/// Days from 1970-01-01 (the algorithm's Unix epoch) to 2000-01-01 (PostgreSQL's date epoch). Used to
/// rebase between the two so `DATE` values are days-since-2000.
const PG_EPOCH_UNIX_DAYS: i64 = 10957;

/// Microseconds per day / per second -- the `timestamp` is i64 microseconds since the PG epoch.
const MICROS_PER_DAY: i64 = 86_400_000_000;
const MICROS_PER_SEC: i64 = 1_000_000;

// PostgreSQL REL_16's `ParseDateTime` work buffers. The timestamp input path gets one extra byte
// for every possible field because every copied token is NUL-terminated independently.
const PG_MAX_DATE_LEN: usize = 128;
const PG_MAX_DATETIME_FIELDS: usize = 25;
const PG_MAX_DATE_WORKSPACE_BYTES: usize = PG_MAX_DATE_LEN + 1;
const PG_MAX_TIMESTAMP_WORKSPACE_BYTES: usize = PG_MAX_DATE_LEN + PG_MAX_DATETIME_FIELDS;

/// PostgreSQL's finite `DATE` binary carrier starts at `DATETIME_MIN_JULIAN` rebased to the
/// PostgreSQL epoch and ends just before `DATE_END_JULIAN`. The adjacent i32 sentinels encode
/// `-infinity`/`infinity`, so accepting every i32 would let a wire value bypass finite SQL input.
pub const PG_DATE_MIN_DAYS: i32 = -2_451_545;
pub const PG_DATE_END_DAYS_EXCLUSIVE: i32 = 2_145_031_949;

/// PostgreSQL's finite `timestamp without time zone` binary carrier. Both endpoints are public
/// because text parsing, binary Bind decoding, and already-typed prepared values must share the
/// same finite-domain proof. The upper endpoint is exclusive (`294277-01-01 00:00:00`).
pub const PG_TIMESTAMP_MIN_MICROS: i64 = -211_813_488_000_000_000;
pub const PG_TIMESTAMP_END_MICROS_EXCLUSIVE: i64 = 9_223_371_331_200_000_000;

/// The precise reason an ISO temporal input could not be represented.
///
/// The public [`parse_date`] and [`parse_timestamp`] compatibility wrappers deliberately still
/// return `Option`; SQL, engine, and wire callers use these detailed outcomes to preserve
/// PostgreSQL's `22007` (bad syntax) versus `22008` (a syntactically numeric field/range overflow)
/// distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatetimeParseError {
    MalformedFormat,
    FieldOverflow,
}

/// Validate a finite PostgreSQL `DATE` binary carrier (days since 2000-01-01).
pub fn validate_date_carrier(days: i32) -> Result<i32, DatetimeParseError> {
    if (PG_DATE_MIN_DAYS..PG_DATE_END_DAYS_EXCLUSIVE).contains(&days) {
        Ok(days)
    } else {
        Err(DatetimeParseError::FieldOverflow)
    }
}

/// Validate a finite PostgreSQL `TIMESTAMP` binary carrier (microseconds since 2000-01-01).
pub fn validate_timestamp_carrier(micros: i64) -> Result<i64, DatetimeParseError> {
    if (PG_TIMESTAMP_MIN_MICROS..PG_TIMESTAMP_END_MICROS_EXCLUSIVE).contains(&micros) {
        Ok(micros)
    } else {
        Err(DatetimeParseError::FieldOverflow)
    }
}

/// An ASCII-digits-only unsigned component (a date/time field), as i64. Rejects an empty string and a
/// leading sign / whitespace that Rust's integer parse would otherwise accept (PG rejects `+2024`).
fn ascii_digits(s: &str) -> Result<i64, DatetimeParseError> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(DatetimeParseError::MalformedFormat);
    }
    // `strtoint` accepts arbitrarily many leading zeroes when the resulting numeric value fits.
    // Trim them before Rust's checked conversion so the same values do not become an artificial
    // overflow merely because PostgreSQL's input workspace permitted a long token.
    let significant = s.trim_start_matches('0');
    if significant.is_empty() {
        return Ok(0);
    }
    significant
        .parse()
        .map_err(|_| DatetimeParseError::FieldOverflow)
}

/// Mirror the byte accounting of PostgreSQL REL_16's `ParseDateTime` before numeric conversion.
///
/// Every copied token consumes its bytes plus a NUL terminator; ASCII whitespace is skipped and
/// `T`/`BC` are separate alpha tokens. This intentionally is not a raw-input-length limit.
fn check_datetime_workspace(text: &str, workspace_bytes: usize) -> Result<(), DatetimeParseError> {
    let bytes = text.as_bytes();
    let mut at = 0;
    let mut used = 0_usize;
    let mut fields = 0_usize;

    while at < bytes.len() {
        if bytes[at].is_ascii_whitespace() {
            at += 1;
            continue;
        }

        let start = at;
        match bytes[at] {
            byte if byte.is_ascii_digit() => {
                at += 1;
                while bytes.get(at).is_some_and(u8::is_ascii_digit) {
                    at += 1;
                }
                if bytes.get(at) == Some(&b':') {
                    at += 1;
                    while bytes
                        .get(at)
                        .is_some_and(|byte| byte.is_ascii_digit() || matches!(*byte, b':' | b'.'))
                    {
                        at += 1;
                    }
                } else if let Some(delimiter @ (b'-' | b'/' | b'.')) = bytes.get(at).copied() {
                    at += 1;
                    if bytes.get(at).is_some_and(u8::is_ascii_digit) {
                        while bytes.get(at).is_some_and(u8::is_ascii_digit) {
                            at += 1;
                        }
                        if bytes.get(at) == Some(&delimiter) {
                            at += 1;
                            while bytes
                                .get(at)
                                .is_some_and(|byte| byte.is_ascii_digit() || *byte == delimiter)
                            {
                                at += 1;
                            }
                        }
                    } else {
                        while bytes
                            .get(at)
                            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == delimiter)
                        {
                            at += 1;
                        }
                    }
                }
            }
            b'.' => {
                at += 1;
                while bytes.get(at).is_some_and(u8::is_ascii_digit) {
                    at += 1;
                }
            }
            byte if byte.is_ascii_alphabetic() => {
                at += 1;
                while bytes.get(at).is_some_and(u8::is_ascii_alphabetic) {
                    at += 1;
                }
            }
            b'+' | b'-' => {
                at += 1;
                while bytes.get(at).is_some_and(u8::is_ascii_whitespace) {
                    at += 1;
                }
                if bytes.get(at).is_some_and(u8::is_ascii_digit) {
                    at += 1;
                    while bytes.get(at).is_some_and(|byte| {
                        byte.is_ascii_digit() || matches!(*byte, b':' | b'.' | b'-')
                    }) {
                        at += 1;
                    }
                } else if bytes.get(at).is_some_and(u8::is_ascii_alphabetic) {
                    at += 1;
                    while bytes.get(at).is_some_and(u8::is_ascii_alphabetic) {
                        at += 1;
                    }
                } else {
                    return Err(DatetimeParseError::MalformedFormat);
                }
            }
            byte if byte.is_ascii_punctuation() => {
                at += 1;
                continue;
            }
            _ => return Err(DatetimeParseError::MalformedFormat),
        }

        fields = fields
            .checked_add(1)
            .filter(|fields| *fields <= PG_MAX_DATETIME_FIELDS)
            .ok_or(DatetimeParseError::MalformedFormat)?;
        used = used
            .checked_add(at - start)
            .and_then(|used| used.checked_add(1))
            .filter(|used| *used <= workspace_bytes)
            .ok_or(DatetimeParseError::MalformedFormat)?;
    }
    Ok(())
}

/// Checked Howard Hinnant civil-date conversion used while accepting external text. The formatter
/// only receives bounded i32 day counts; a parser must reject an enormous numeric year without
/// wrapping in debug or release builds.
fn days_from_civil_checked(y: i64, m: i64, d: i64) -> Option<i64> {
    let y = if m <= 2 { y.checked_sub(1)? } else { y };
    let era = if y >= 0 {
        y / 400
    } else {
        y.checked_sub(399)? / 400
    };
    let yoe = y.checked_sub(era.checked_mul(400)?)?; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // Mar=0 .. Feb=11
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe
        .checked_mul(365)?
        .checked_add(yoe / 4)?
        .checked_sub(yoe / 100)?
        .checked_add(doy)?; // [0, 146096]
    era.checked_mul(146097)?
        .checked_add(doe)?
        .checked_sub(719468)
}

/// The civil date `(year, month, day)` for a count of days since 1970-01-01 (the inverse of
/// [`days_from_civil`], Howard Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Separate the PostgreSQL ISO `BC` suffix from a date or timestamp literal.
///
/// The suffix is ASCII-case-insensitive and may immediately follow the preceding field, matching
/// PostgreSQL's `4714-11-24bc` acceptance as well as the canonical output's `4714-11-24 BC`.
/// Other era spellings and placements deliberately remain outside this narrow ISO grammar.
fn split_bc_suffix(text: &str) -> Result<(&str, bool), DatetimeParseError> {
    let text = text.trim();
    let Some(suffix) = text.get(text.len().saturating_sub(2)..) else {
        return Err(DatetimeParseError::MalformedFormat);
    };
    if !suffix.eq_ignore_ascii_case("BC") {
        return Ok((text, false));
    }
    let body = text[..text.len() - suffix.len()].trim_end();
    if body.is_empty() {
        return Err(DatetimeParseError::MalformedFormat);
    }
    Ok((body, true))
}

/// Numeric ISO date fields before calendar/range validation.
///
/// PostgreSQL's `DecodeDate` converts numeric fields while `DecodeDateTime` walks its token list,
/// but defers `ValidateDate` until every later field has been decoded. Keeping that split is
/// observable when a bad date is followed by a bad time or an unsupported trailing token.
#[derive(Debug, Clone, Copy)]
struct UncheckedCivilDate {
    year: i64,
    month: i64,
    day: i64,
    bc: bool,
}

/// Parse ISO date field syntax and numeric conversion, deliberately deferring calendar and carrier
/// validation to [`validate_iso_date_fields`].
fn parse_iso_date_fields(text: &str, bc: bool) -> Result<UncheckedCivilDate, DatetimeParseError> {
    let mut parts = text.splitn(3, '-');
    Ok(UncheckedCivilDate {
        year: ascii_digits(parts.next().ok_or(DatetimeParseError::MalformedFormat)?)?,
        month: ascii_digits(parts.next().ok_or(DatetimeParseError::MalformedFormat)?)?,
        day: ascii_digits(parts.next().ok_or(DatetimeParseError::MalformedFormat)?)?,
        bc,
    })
}

/// Apply PostgreSQL's deferred `ValidateDate`-equivalent checks to already decoded ISO fields.
fn validate_iso_date_fields(date: UncheckedCivilDate) -> Result<i32, DatetimeParseError> {
    let UncheckedCivilDate {
        year,
        month,
        day,
        bc,
    } = date;
    if year == 0 || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(DatetimeParseError::FieldOverflow);
    }
    // PostgreSQL's no-year-zero display convention maps `1 BC` to astronomical year zero.
    let astronomical_y = if bc { 1 - year } else { year };
    let days = days_from_civil_checked(astronomical_y, month, day)
        .and_then(|days| days.checked_sub(PG_EPOCH_UNIX_DAYS))
        .ok_or(DatetimeParseError::FieldOverflow)?;
    let days = i32::try_from(days).map_err(|_| DatetimeParseError::FieldOverflow)?;
    // Round-trip to reject a day that does not exist in that month (e.g. 2024-02-30 -> 2024-03-01).
    if civil_from_days(i64::from(days) + PG_EPOCH_UNIX_DAYS) != (astronomical_y, month, day) {
        return Err(DatetimeParseError::FieldOverflow);
    }
    validate_date_carrier(days)
}

/// Parse a PostgreSQL `DATE` literal in ISO `YYYY-MM-DD[ BC]` form to i32 days since 2000-01-01
/// with a typed failure. Leading/trailing ASCII whitespace is tolerated. Other non-ISO styles and
/// negative years are not accepted yet (a follow-on).
pub fn parse_date_detailed(text: &str) -> Result<i32, DatetimeParseError> {
    check_datetime_workspace(text, PG_MAX_DATE_WORKSPACE_BYTES)?;
    let (date, bc) = split_bc_suffix(text)?;
    validate_iso_date_fields(parse_iso_date_fields(date, bc)?)
}

/// Option-returning compatibility wrapper around [`parse_date_detailed`].
pub fn parse_date(text: &str) -> Option<i32> {
    parse_date_detailed(text).ok()
}

/// Format i32 days since 2000-01-01 as the ISO `YYYY-MM-DD` text PostgreSQL emits.
/// Astronomical years at or before zero are rendered in PostgreSQL's `N BC` notation.
pub fn format_date(days: i32) -> String {
    let (y, m, d) = civil_from_days(i64::from(days) + PG_EPOCH_UNIX_DAYS);
    format_civil_date(y, m, d)
}

fn format_civil_date(y: i64, m: i64, d: i64) -> String {
    if y <= 0 {
        format!("{:04}-{m:02}-{d:02} BC", 1 - y)
    } else {
        format!("{y:04}-{m:02}-{d:02}")
    }
}

/// The timestamp time-tail field retained after narrow ISO separator decoding.
enum IsoTimestampTimePart<'a> {
    Absent,
    Text(&'a str),
    EmptyAdjacentT,
}

/// Split the narrow ISO timestamp form at its first space or `T` separator.
///
/// The surrounding workspace checker has already applied `ParseDateTime`'s token budget. This
/// preserves the existing narrow grammar's `T` behavior while leaving the time tail available for
/// left-to-right DecodeDateTime-style processing.
fn split_iso_timestamp_parts(text: &str) -> (&str, IsoTimestampTimePart<'_>) {
    let Some(separator) = text
        .as_bytes()
        .iter()
        .position(|byte| byte.is_ascii_whitespace() || *byte == b'T')
    else {
        return (text, IsoTimestampTimePart::Absent);
    };
    let separator_is_t = text.as_bytes()[separator] == b'T';
    let time = text[separator + 1..].trim();
    if separator_is_t && time.is_empty() {
        // This is a later ParseDateTime field, so defer its 22007 until DecodeDate has performed
        // date-field syntax/numeric conversion. Calendar validation remains deferred further.
        return (
            text[..separator].trim_end(),
            IsoTimestampTimePart::EmptyAdjacentT,
        );
    }
    if time.is_empty() {
        (text[..separator].trim_end(), IsoTimestampTimePart::Absent)
    } else {
        (
            text[..separator].trim_end(),
            IsoTimestampTimePart::Text(time),
        )
    }
}

/// Return the first numeric datetime token exactly as the narrow ParseDateTime grammar admits it:
/// an initial digit followed by digits, colons, and decimal points. Any remaining bytes represent
/// later fields and are deliberately handled only after the decoded time has had a chance to
/// report its range error.
fn split_datetime_time_token(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    if !bytes.first().is_some_and(u8::is_ascii_digit) {
        return None;
    }
    let mut end = 1;
    while bytes
        .get(end)
        .is_some_and(|byte| byte.is_ascii_digit() || matches!(*byte, b':' | b'.'))
    {
        end += 1;
    }
    Some((&text[..end], &text[end..]))
}

/// Parse a time-of-day `HH:MM[:SS][.ffffff]` to microseconds within the day.
///
/// Without seconds, PostgreSQL treats `HH:MM` as hours and minutes. With a decimal point, however,
/// its `DecodeTimeCommon` treats exactly two components as `MM:SS.fraction`: `10:20.5` is ten
/// minutes, twenty and a half seconds. PostgreSQL parses the decimal fraction through `strtod`,
/// then applies `rint` to microseconds. We intentionally mirror that binary-float, ties-to-even
/// behavior rather than decimal rounding. A leap-second spelling (`HH:MM:60`) accepts a normally
/// rounded fraction and normalizes forward by one second only while the resulting time does not
/// exceed the end of its day; `24:00[:00[.0...]]` accepts only a fraction that rounds to zero and
/// normalizes to the next midnight.
fn parse_time_of_day_detailed(text: &str) -> Result<i64, DatetimeParseError> {
    let (hms, fraction) = match text.find('.') {
        Some(dot) => (&text[..dot], Some(&text[dot..])),
        None => (text, None),
    };
    let mut parts = hms.split(':');
    let first = ascii_digits(parts.next().ok_or(DatetimeParseError::MalformedFormat)?)?;
    let second = ascii_digits(parts.next().ok_or(DatetimeParseError::MalformedFormat)?)?;
    let third = match parts.next() {
        Some(sec) => Some(ascii_digits(sec)?),
        None => None,
    };
    if parts.next().is_some() {
        return Err(DatetimeParseError::MalformedFormat);
    }
    let (h, m, s) = match (third, fraction.is_some()) {
        // REL_16 DecodeTimeCommon's minute-to-second fractional shorthand.
        (None, true) => (0, first, second),
        (None, false) => (first, second, 0),
        (Some(seconds), _) => (first, second, seconds),
    };
    let frac_micros = match fraction {
        None | Some(".") => 0,
        Some(fraction) => {
            let digits = &fraction[1..];
            if !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(DatetimeParseError::MalformedFormat);
            }
            // REL_16 ParseFractionalSecond: `strtod(".<digits>")`, followed by
            // `rint(frac * 1000000)`. Keep the original dot-prefixed input borrowed so a long
            // fraction cannot turn into an unbounded temporary allocation before the ERANGE
            // guards below run.
            let fraction = fraction
                .parse::<f64>()
                .map_err(|_| DatetimeParseError::MalformedFormat)?;
            if !fraction.is_finite()
                || (fraction == 0.0 && digits.bytes().any(|digit| digit != b'0'))
                // `strtod` reports ERANGE for nonzero subnormals too. Rust exposes the value but
                // not errno, so reject that same observable conversion class explicitly.
                || (fraction > 0.0 && fraction < f64::MIN_POSITIVE)
            {
                return Err(DatetimeParseError::MalformedFormat);
            }
            let micros = (fraction * MICROS_PER_SEC as f64).round_ties_even();
            if !micros.is_finite() || !(0.0..=MICROS_PER_SEC as f64).contains(&micros) {
                return Err(DatetimeParseError::MalformedFormat);
            }
            micros as i64
        }
    };
    if h > 24 || m > 59 || s > 60 {
        return Err(DatetimeParseError::FieldOverflow);
    }
    let total = h
        .checked_mul(3_600_000_000)
        .and_then(|total| total.checked_add(m * 60_000_000))
        .and_then(|total| total.checked_add(s * MICROS_PER_SEC))
        .and_then(|total| total.checked_add(frac_micros))
        .ok_or(DatetimeParseError::FieldOverflow)?;
    (total <= MICROS_PER_DAY)
        .then_some(total)
        .ok_or(DatetimeParseError::FieldOverflow)
}

/// Parse a PostgreSQL `TIMESTAMP` literal `YYYY-MM-DD[ T]HH:MM[:SS[.ffffff]][ BC]` (the time part
/// optional, defaulting to midnight) to i64 MICROSECONDS since 2000-01-01 00:00:00 with a typed
/// failure. The BC suffix follows the optional time/fraction. Non-ISO styles and time zones are
/// not accepted yet (a follow-on).
pub fn parse_timestamp_detailed(text: &str) -> Result<i64, DatetimeParseError> {
    check_datetime_workspace(text, PG_MAX_TIMESTAMP_WORKSPACE_BYTES)?;
    let (trimmed, bc) = split_bc_suffix(text)?;
    let (date_part, time_part) = split_iso_timestamp_parts(trimmed);
    // DecodeDate converts numeric fields now, but ValidateDate waits until all later fields have
    // been processed. In particular, an out-of-range time takes precedence over a bad calendar
    // date, while a later unsupported field takes precedence over that deferred date validation.
    let date = parse_iso_date_fields(date_part, bc)?;
    let tod = match time_part {
        IsoTimestampTimePart::EmptyAdjacentT => return Err(DatetimeParseError::MalformedFormat),
        IsoTimestampTimePart::Text(t) => {
            let Some((time_token, remainder)) = split_datetime_time_token(t) else {
                return Err(DatetimeParseError::MalformedFormat);
            };
            let tod = parse_time_of_day_detailed(time_token)?;
            if !remainder.trim().is_empty() {
                return Err(DatetimeParseError::MalformedFormat);
            }
            tod
        }
        IsoTimestampTimePart::Absent => 0,
    };
    let days = i64::from(validate_iso_date_fields(date)?);
    let micros = days
        .checked_mul(MICROS_PER_DAY)
        .and_then(|micros| micros.checked_add(tod))
        .ok_or(DatetimeParseError::FieldOverflow)?;
    validate_timestamp_carrier(micros)
}

/// Option-returning compatibility wrapper around [`parse_timestamp_detailed`].
pub fn parse_timestamp(text: &str) -> Option<i64> {
    parse_timestamp_detailed(text).ok()
}

/// Format i64 microseconds since 2000-01-01 00:00:00 as the text PostgreSQL emits: `YYYY-MM-DD
/// HH:MM:SS`, plus a `.ffffff` fractional part (trailing zeros trimmed) when non-zero. Astronomical
/// years at or before zero use a trailing ` BC` marker.
pub fn format_timestamp(micros: i64) -> String {
    let days = micros.div_euclid(MICROS_PER_DAY);
    let tod = micros.rem_euclid(MICROS_PER_DAY);
    let (y, mo, d) = civil_from_days(days + PG_EPOCH_UNIX_DAYS);
    let secs = tod / MICROS_PER_SEC;
    let frac = tod % MICROS_PER_SEC;
    let (h, mi, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    let date = if y <= 0 {
        format!("{:04}-{mo:02}-{d:02}", 1 - y)
    } else {
        format!("{y:04}-{mo:02}-{d:02}")
    };
    let mut out = format!("{date} {h:02}:{mi:02}:{s:02}");
    if frac != 0 {
        out.push('.');
        out.push_str(format!("{frac:06}").trim_end_matches('0'));
    }
    if y <= 0 {
        out.push_str(" BC");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_epoch_and_neighbors() {
        assert_eq!(
            parse_date("2000-01-01"),
            Some(0),
            "the PG date epoch is day 0"
        );
        assert_eq!(parse_date("2000-01-02"), Some(1));
        assert_eq!(parse_date("1999-12-31"), Some(-1));
        assert_eq!(format_date(0), "2000-01-01");
        assert_eq!(format_date(-1), "1999-12-31");
    }

    #[test]
    fn date_round_trips_across_centuries() {
        // Every day over ~80 years parses and formats back to the same ISO text, and consecutive
        // dates are consecutive day numbers (monotone, no gaps) -- a closed-form oracle.
        let mut prev = parse_date("1970-01-01").expect("1970-01-01");
        for days in (parse_date("1970-01-02").unwrap())..=(parse_date("2050-12-31").unwrap()) {
            let text = format_date(days);
            assert_eq!(parse_date(&text), Some(days), "round-trip {text}");
            assert_eq!(days, prev + 1, "consecutive days increment by 1 at {text}");
            prev = days;
        }
    }

    #[test]
    fn date_leap_year_rules() {
        assert!(parse_date("2024-02-29").is_some(), "2024 is a leap year");
        assert_eq!(parse_date("2023-02-29"), None, "2023 is not a leap year");
        assert_eq!(
            parse_date("2000-02-29"),
            parse_date("2000-02-29"),
            "2000 IS a leap year"
        );
        assert!(
            parse_date("2000-02-29").is_some(),
            "2000 is divisible by 400 -> leap"
        );
        assert_eq!(
            parse_date("1900-02-29"),
            None,
            "1900 is divisible by 100 not 400 -> not leap"
        );
    }

    #[test]
    fn date_rejects_malformed() {
        assert_eq!(parse_date("2024-13-01"), None, "month 13");
        assert_eq!(parse_date("2024-00-01"), None, "month 0");
        assert_eq!(parse_date("2024-02-30"), None, "Feb 30 does not exist");
        assert_eq!(parse_date("2024-04-31"), None, "April has 30 days");
        assert_eq!(parse_date("not-a-date"), None);
        assert_eq!(parse_date("2024-01"), None, "missing day");
        assert_eq!(parse_date("2024-01-15-extra"), None, "trailing junk");
        assert_eq!(parse_date(""), None);
        // PG rejects a leading sign on a component (Rust's int parse would otherwise accept it).
        assert_eq!(parse_date("+2024-01-15"), None, "leading + on the year");
        assert_eq!(parse_date("2024-+01-15"), None, "leading + on the month");
        assert_eq!(parse_date("2024-01-+15"), None, "leading + on the day");
        assert_eq!(
            parse_date("2024- 01-15"),
            None,
            "embedded space in a component"
        );
    }

    #[test]
    fn bc_dates_and_timestamps_round_trip_through_postgresql_output() {
        for (input, expected) in [
            ("4714-11-24 BC", "4714-11-24 BC"),
            ("4714-11-24bc", "4714-11-24 BC"),
            ("0001-01-01 BC", "0001-01-01 BC"),
            ("0044-03-15 bC", "0044-03-15 BC"),
        ] {
            let days =
                parse_date_detailed(input).unwrap_or_else(|error| panic!("{input}: {error:?}"));
            assert_eq!(format_date(days), expected, "{input}");
            assert_eq!(parse_date_detailed(&format_date(days)), Ok(days));
        }
        assert_eq!(parse_date_detailed("4714-11-24 BC"), Ok(PG_DATE_MIN_DAYS));

        for (input, expected) in [
            ("4714-11-24 00:00:00 BC", "4714-11-24 00:00:00 BC"),
            (
                "0044-03-15T12:34:56.123456 bC",
                "0044-03-15 12:34:56.123456 BC",
            ),
            ("0001-01-01 BC", "0001-01-01 00:00:00 BC"),
        ] {
            let micros = parse_timestamp_detailed(input)
                .unwrap_or_else(|error| panic!("{input}: {error:?}"));
            assert_eq!(format_timestamp(micros), expected, "{input}");
            assert_eq!(
                parse_timestamp_detailed(&format_timestamp(micros)),
                Ok(micros)
            );
        }
        assert_eq!(
            parse_timestamp_detailed("4714-11-24 00:00:00 BC"),
            Ok(PG_TIMESTAMP_MIN_MICROS)
        );
        assert_eq!(
            parse_timestamp_detailed("4714-11-24 00:00:00.0000005 BC"),
            Ok(PG_TIMESTAMP_MIN_MICROS),
            "the lower carrier retains PostgreSQL's binary-float round-to-even result"
        );
    }

    #[test]
    fn bc_era_syntax_and_range_errors_keep_temporal_taxonomy() {
        for text in ["BC", "4714-11-24 B", "4714-11-24 BCE"] {
            assert_eq!(
                parse_date_detailed(text),
                Err(DatetimeParseError::MalformedFormat),
                "{text}"
            );
        }
        for text in ["0000-01-01 BC", "4714-11-23 BC"] {
            assert_eq!(
                parse_date_detailed(text),
                Err(DatetimeParseError::FieldOverflow),
                "{text}"
            );
            assert_eq!(parse_date(text), None, "Option wrapper {text}");
        }
        for text in [
            "4714-11-24 BC 00:00:00",
            "4714-11-24 00:00:00 BCE",
            "4714-11-24 00:00:00 BC BC",
        ] {
            assert_eq!(
                parse_timestamp_detailed(text),
                Err(DatetimeParseError::MalformedFormat),
                "{text}"
            );
        }
        for text in ["0000-01-01 00:00:00 BC", "4714-11-23 00:00:00 BC"] {
            assert_eq!(
                parse_timestamp_detailed(text),
                Err(DatetimeParseError::FieldOverflow),
                "{text}"
            );
            assert_eq!(parse_timestamp(text), None, "Option wrapper {text}");
        }
        let underflow = format!("2024-01-01 00:00:00.{}1", "0".repeat(400));
        assert_eq!(
            parse_timestamp_detailed(&underflow),
            Err(DatetimeParseError::MalformedFormat),
            "the oversized fractional token is rejected by ParseDateTime workspace accounting"
        );
        let subnormal = format!("2024-01-01 00:00:00.{}1", "0".repeat(308));
        assert_eq!(
            parse_timestamp_detailed(&subnormal),
            Err(DatetimeParseError::MalformedFormat),
            "the oversized fractional token is rejected by ParseDateTime workspace accounting"
        );
    }

    #[test]
    fn timestamp_epoch_and_units() {
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00"),
            Some(0),
            "the timestamp epoch is micro 0"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:01"),
            Some(1_000_000),
            "one second"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:01:00"),
            Some(60_000_000),
            "one minute"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 01:00:00"),
            Some(3_600_000_000),
            "one hour"
        );
        assert_eq!(
            parse_timestamp("2000-01-02 00:00:00"),
            Some(MICROS_PER_DAY),
            "one day"
        );
        // no time part -> midnight; 'T' separator accepted; HH:MM defaults seconds to 0.
        assert_eq!(
            parse_timestamp("2000-01-02"),
            Some(MICROS_PER_DAY),
            "date-only => midnight"
        );
        assert_eq!(
            parse_timestamp("2000-01-01T00:01:00"),
            Some(60_000_000),
            "'T' separator"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:01"),
            Some(60_000_000),
            "HH:MM (no seconds)"
        );
    }

    #[test]
    fn timestamp_fractional_seconds() {
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.5"),
            Some(500_000),
            ".5 => 500000 us"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.000001"),
            Some(1),
            "one microsecond"
        );
        assert_eq!(parse_timestamp("2000-01-01 00:00:00.123456"), Some(123_456));
        // PostgreSQL applies rint ties-to-even to a binary `strtod` result, so decimal-looking
        // midpoints can land on either side of the binary midpoint.
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.1234564"),
            Some(123_456),
            "round down"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.1234565"),
            Some(123_456),
            "exact tie retains an even microsecond"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.1234575"),
            Some(123_458),
            "exact tie advances an odd microsecond"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.1234567"),
            Some(123_457),
            "round up"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.0000005"),
            Some(0),
            "an exact zero tie stays even"
        );
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.0000005000001"),
            Some(1),
            "digits beyond an exact tie round up"
        );
        for (fraction, expected) in [
            (".0001255", 125),
            (".0001265", 127),
            (".0010005", 1_001),
            (".0001255000000000000000", 125),
            (".0001255000000000000001", 125),
            (".0001265000000000000000", 127),
            (".0001265000000000000001", 127),
        ] {
            assert_eq!(
                parse_timestamp(&format!("2000-01-01 00:00:00{fraction}")),
                Some(expected),
                "PostgreSQL strtod/rint matrix {fraction}"
            );
        }
        // Rounding that carries: .9999995 -> 1_000_000 us = one second past the epoch.
        assert_eq!(
            parse_timestamp("2000-01-01 00:00:00.9999995"),
            Some(1_000_000),
            "carry into seconds"
        );
        // Carry across the end of a day rolls into the next day's midnight.
        assert_eq!(
            parse_timestamp("2024-01-15 23:59:59.9999995"),
            parse_timestamp("2024-01-16 00:00:00"),
            "carry across the day boundary"
        );
        // PG trims trailing zeros from the fractional part on output, omits it when zero.
        assert_eq!(format_timestamp(0), "2000-01-01 00:00:00");
        assert_eq!(format_timestamp(500_000), "2000-01-01 00:00:00.5");
        assert_eq!(format_timestamp(123_456), "2000-01-01 00:00:00.123456");
    }

    #[test]
    fn fraction_conversion_rejects_strtod_erange_classes_without_workspace_guard() {
        for (zeros, description) in [
            (400, "nonzero fractional underflow to zero"),
            (308, "nonzero fractional subnormal"),
        ] {
            let time = format!("00:00:00.{}1", "0".repeat(zeros));
            assert_eq!(
                parse_time_of_day_detailed(&time),
                Err(DatetimeParseError::MalformedFormat),
                "{description}"
            );
        }
    }

    #[test]
    fn pg16_workspace_limits_precede_numeric_date_and_timestamp_parsing() {
        let space_timestamp = format!("2000-01-01 00:00:00.5{}", "0".repeat(131));
        assert_eq!(space_timestamp.len(), 152);
        assert_eq!(parse_timestamp_detailed(&space_timestamp), Ok(500_000));
        let space_timestamp_over = format!("2000-01-01 00:00:00.5{}", "0".repeat(132));
        assert_eq!(space_timestamp_over.len(), 153);
        assert_eq!(
            parse_timestamp_detailed(&space_timestamp_over),
            Err(DatetimeParseError::MalformedFormat)
        );

        let t_timestamp = format!("2000-01-01T00:00:00.5{}", "0".repeat(129));
        assert_eq!(t_timestamp.len(), 150);
        assert_eq!(parse_timestamp_detailed(&t_timestamp), Ok(500_000));
        let t_timestamp_over = format!("2000-01-01T00:00:00.5{}", "0".repeat(130));
        assert_eq!(t_timestamp_over.len(), 151);
        assert_eq!(
            parse_timestamp_detailed(&t_timestamp_over),
            Err(DatetimeParseError::MalformedFormat)
        );

        let bc_timestamp = format!("2000-01-01 00:00:00.5{} BC", "0".repeat(128));
        assert_eq!(bc_timestamp.len(), 152);
        assert!(parse_timestamp_detailed(&bc_timestamp).is_ok());
        let bc_timestamp_over = format!("2000-01-01 00:00:00.5{} BC", "0".repeat(129));
        assert_eq!(bc_timestamp_over.len(), 153);
        assert_eq!(
            parse_timestamp_detailed(&bc_timestamp_over),
            Err(DatetimeParseError::MalformedFormat)
        );

        let date = format!("{}2000-01-01", "0".repeat(118));
        assert_eq!(date.len(), 128);
        assert_eq!(parse_date_detailed(&date), Ok(0));
        let date_over = format!("{}2000-01-01", "0".repeat(119));
        assert_eq!(date_over.len(), 129);
        assert_eq!(
            parse_date_detailed(&date_over),
            Err(DatetimeParseError::MalformedFormat)
        );

        let overflowing_date = format!("{}-01-01", "9".repeat(122));
        assert_eq!(overflowing_date.len(), 128);
        assert_eq!(
            parse_date_detailed(&overflowing_date),
            Err(DatetimeParseError::FieldOverflow)
        );
        let overflowing_date_over = format!("{}-01-01", "9".repeat(123));
        assert_eq!(overflowing_date_over.len(), 129);
        assert_eq!(
            parse_date_detailed(&overflowing_date_over),
            Err(DatetimeParseError::MalformedFormat)
        );

        let timestamp_date_only = format!("{}2000-01-01", "0".repeat(142));
        assert_eq!(timestamp_date_only.len(), 152);
        assert_eq!(parse_timestamp_detailed(&timestamp_date_only), Ok(0));
        let timestamp_date_only_over = format!("{}2000-01-01", "0".repeat(143));
        assert_eq!(timestamp_date_only_over.len(), 153);
        assert_eq!(
            parse_timestamp_detailed(&timestamp_date_only_over),
            Err(DatetimeParseError::MalformedFormat)
        );

        let padded = format!(
            "{}2000-01-01{}00:00:00.5{}",
            " ".repeat(200),
            " ".repeat(200),
            " ".repeat(200)
        );
        assert_eq!(parse_timestamp_detailed(&padded), Ok(500_000));
    }

    #[test]
    fn pg16_decode_time_common_accepts_bare_and_two_field_fractions() {
        for (text, expected) in [
            ("2000-01-01 00:00:00.", 0),
            ("2000-01-01 10:20.", 620_000_000),
            ("2000-01-01 10:20.5", 620_500_000),
            ("2000-01-01 23:59.5", 1_439_500_000),
            ("2000-01-01 23:59", 86_340_000_000),
            ("2000-01-01 59:60.999999", 3_600_999_999),
        ] {
            assert_eq!(parse_timestamp_detailed(text), Ok(expected), "{text}");
        }
        for text in ["2000-01-01 60:00.5", "2000-01-01 59:61.5"] {
            assert_eq!(
                parse_timestamp_detailed(text),
                Err(DatetimeParseError::FieldOverflow),
                "{text}"
            );
        }
    }

    #[test]
    fn timestamp_round_trips_incl_before_epoch() {
        for s in [
            "2024-01-15 10:30:45",
            "2024-12-31 23:59:59",
            "2000-01-01 00:00:00",
            "1999-12-31 12:00:00", // before the epoch (negative micros)
            "1970-01-01 00:00:00",
            "2024-06-15 08:09:10.123456",
        ] {
            let micros = parse_timestamp(s).unwrap_or_else(|| panic!("parse {s}"));
            assert_eq!(format_timestamp(micros), s, "round-trip {s}");
        }
        // a full day-by-day + noon sweep stays monotone in micros.
        let mut prev = parse_timestamp("2020-01-01 12:00:00").unwrap();
        for day in 1..400 {
            let micros = prev + MICROS_PER_DAY;
            assert_eq!(
                parse_timestamp(&format_timestamp(micros)),
                Some(micros),
                "round-trip day {day}"
            );
            assert_eq!(micros, prev + MICROS_PER_DAY);
            prev = micros;
        }
    }

    #[test]
    fn timestamp_rejects_malformed() {
        assert_eq!(parse_timestamp("2024-01-15 25:00:00"), None, "hour 25");
        assert_eq!(parse_timestamp("2024-01-15 10:60:00"), None, "minute 60");
        assert_eq!(
            parse_timestamp("2024-01-15 10:00:60"),
            parse_timestamp("2024-01-15 10:01:00"),
            "leap second normalizes into the next minute"
        );
        assert_eq!(
            parse_timestamp("2024-13-01 10:00:00"),
            None,
            "bad date part"
        );
        assert_eq!(
            parse_timestamp("2024-01-15 10:00:00:00"),
            None,
            "too many time parts"
        );
        assert_eq!(
            parse_timestamp("2024-01-15 10:xx:00"),
            None,
            "non-numeric minute"
        );
        assert_eq!(
            parse_timestamp("2024-01-15 10:00:00.abc"),
            None,
            "non-numeric fraction"
        );
    }

    #[test]
    fn detailed_temporal_parse_preserves_wrappers_and_sqlstate_taxonomy() {
        for text in ["not-a-date", "2024-01", "2024-01-15 10:xx:00"] {
            let detailed = if text.contains(':') {
                parse_timestamp_detailed(text).map(|_| ())
            } else {
                parse_date_detailed(text).map(|_| ())
            };
            assert_eq!(detailed, Err(DatetimeParseError::MalformedFormat), "{text}");
        }

        for text in [
            "0000-01-01",
            "2024-13-01",
            "2024-02-30",
            "6000000-01-01",
            "999999999999999999999-01-01",
        ] {
            assert_eq!(
                parse_date_detailed(text),
                Err(DatetimeParseError::FieldOverflow),
                "{text}"
            );
            assert_eq!(parse_date(text), None, "Option wrapper {text}");
        }
        for text in [
            "2024-02-30 00:00:00",
            "2024-01-01 25:00:00",
            "2024-01-01 10:60:00",
            "294277-01-01 00:00:00",
            "294277-12-31 00:00:00",
        ] {
            assert_eq!(
                parse_timestamp_detailed(text),
                Err(DatetimeParseError::FieldOverflow),
                "{text}"
            );
            assert_eq!(parse_timestamp(text), None, "Option wrapper {text}");
        }
        assert_eq!(
            parse_date_detailed("2024-02-29"),
            Ok(parse_date("2024-02-29").expect("valid date wrapper"))
        );
        assert_eq!(
            parse_timestamp_detailed("2024-02-29 23:59:59.9999995"),
            Ok(parse_timestamp("2024-03-01 00:00:00").expect("valid timestamp wrapper"))
        );
        assert!(
            parse_timestamp_detailed("294276-12-31 23:59:59.999999").is_ok(),
            "PostgreSQL's last finite timestamp remains representable"
        );
        assert_eq!(
            parse_timestamp_detailed("294276-12-31 23:59:59.9999995"),
            Err(DatetimeParseError::FieldOverflow),
            "rounding across PostgreSQL's exclusive upper bound is rejected"
        );
    }

    #[test]
    fn pg16_timestamp_field_decode_precedence_matches_later_tokens_and_date_validation() {
        // ParseDateTime produces the numeric time token before the trailing alpha field. DecodeTime
        // range checks that token immediately, so those errors win over later unsupported input.
        for text in [
            "2000-01-01 25:00:00.abc",
            "2000-01-01 25:00:00.1x",
            "2000-01-01 10:60:00.abc",
            "2000-01-01 10:00:61.abc",
            "2000-01-01 60:00.abc",
            "2000-01-01 59:61.abc",
            "2000-02-30 25:00:00",
        ] {
            assert_eq!(
                parse_timestamp_detailed(text),
                Err(DatetimeParseError::FieldOverflow),
                "time decode precedes later tokens and deferred date validation: {text}"
            );
        }

        // A valid time lets a later unsupported field win over the deferred calendar validation.
        for text in [
            "2000-02-30 00:00:00.abc",
            "2000-02-30 junk",
            "2000-13-01 junk",
            "4714-11-23 junk BC",
            "2000-01-01 00:00:00.abc",
            // The repeated dot is inside the numeric time token, so DecodeTimeCommon's syntax
            // rejection occurs before it reaches the otherwise out-of-range field check.
            "2000-01-01 25:00:00..abc",
            "2000-01-01 60:00..abc",
        ] {
            assert_eq!(
                parse_timestamp_detailed(text),
                Err(DatetimeParseError::MalformedFormat),
                "later unsupported token precedes deferred date validation: {text}"
            );
        }

        assert_eq!(
            parse_timestamp_detailed("2000-02-30 00:00:00"),
            Err(DatetimeParseError::FieldOverflow),
            "once all fields decode, calendar validation still reports 22008"
        );
    }

    #[test]
    fn pg16_adjacent_trailing_t_defers_its_format_error_until_date_numeric_conversion() {
        let overflowing_dates = [
            format!("{}-01-01", "9".repeat(21)),
            format!("2000-{}-01", "9".repeat(21)),
            format!("2000-01-{}", "9".repeat(21)),
        ];
        for date in &overflowing_dates {
            for suffix in ["T", "TBC"] {
                let text = format!("{date}{suffix}");
                assert_eq!(
                    parse_timestamp_detailed(&text),
                    Err(DatetimeParseError::FieldOverflow),
                    "numeric date overflow precedes adjacent trailing {suffix}: {text}"
                );
            }
            let spaced = format!("{date} T");
            assert_eq!(
                parse_timestamp_detailed(&spaced),
                Err(DatetimeParseError::FieldOverflow),
                "the spaced-T control still decodes the date first: {spaced}"
            );
        }

        for text in ["2000-01-01T", "2000-02-30T", "2000-01-01 T", "2000-02-30 T"] {
            assert_eq!(
                parse_timestamp_detailed(text),
                Err(DatetimeParseError::MalformedFormat),
                "the trailing T is 22007 before deferred calendar validation: {text}"
            );
        }
    }

    #[test]
    fn finite_postgresql_carriers_keep_exact_binary_edges() {
        for days in [PG_DATE_MIN_DAYS, PG_DATE_END_DAYS_EXCLUSIVE - 1] {
            assert_eq!(validate_date_carrier(days), Ok(days));
        }
        for days in [PG_DATE_MIN_DAYS - 1, PG_DATE_END_DAYS_EXCLUSIVE] {
            assert_eq!(
                validate_date_carrier(days),
                Err(DatetimeParseError::FieldOverflow)
            );
        }
        for micros in [
            PG_TIMESTAMP_MIN_MICROS,
            PG_TIMESTAMP_END_MICROS_EXCLUSIVE - 1,
        ] {
            assert_eq!(validate_timestamp_carrier(micros), Ok(micros));
        }
        for micros in [
            PG_TIMESTAMP_MIN_MICROS - 1,
            PG_TIMESTAMP_END_MICROS_EXCLUSIVE,
        ] {
            assert_eq!(
                validate_timestamp_carrier(micros),
                Err(DatetimeParseError::FieldOverflow)
            );
        }
        assert_eq!(format_date(PG_DATE_MIN_DAYS), "4714-11-24 BC");
        assert_eq!(
            format_timestamp(PG_TIMESTAMP_MIN_MICROS),
            "4714-11-24 00:00:00 BC"
        );
    }

    #[test]
    fn end_of_day_requires_zero_fraction_but_leap_seconds_carry_without_exceeding_it() {
        let next_midnight = parse_timestamp_detailed("2024-01-02 00:00:00");
        for text in [
            "2024-01-01 24:00",
            "2024-01-01 24:00:00",
            "2024-01-01 24:00:00.0000000",
            "2024-01-01 24:00:00.0000005",
        ] {
            assert_eq!(parse_timestamp_detailed(text), next_midnight, "{text}");
        }
        for text in [
            "2024-01-01 23:59:60",
            "2024-01-01 23:59:60.0000000",
            "2024-01-01 23:59:60.0000005",
        ] {
            assert_eq!(parse_timestamp_detailed(text), next_midnight, "{text}");
        }
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 10:00:60"),
            parse_timestamp_detailed("2024-01-01 10:01:00")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 10:00:60.0000005"),
            parse_timestamp_detailed("2024-01-01 10:01:00")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 10:00:60.0000005000001"),
            parse_timestamp_detailed("2024-01-01 10:01:00.000001")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 10:00:60.1234565"),
            parse_timestamp_detailed("2024-01-01 10:01:00.123456")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 10:00:60.1234575"),
            parse_timestamp_detailed("2024-01-01 10:01:00.123458")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 10:00:60.0001255"),
            parse_timestamp_detailed("2024-01-01 10:01:00.000125")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 10:00:60.0001265"),
            parse_timestamp_detailed("2024-01-01 10:01:00.000127")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 23:58:60"),
            parse_timestamp_detailed("2024-01-01 23:59:00")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 23:58:60.999999"),
            parse_timestamp_detailed("2024-01-01 23:59:00.999999")
        );
        assert_eq!(
            parse_timestamp_detailed("2024-01-01 23:58:60.9999995"),
            parse_timestamp_detailed("2024-01-01 23:59:01")
        );
        for (text, expected) in [
            ("2024-01-01 24:00:00.1", DatetimeParseError::FieldOverflow),
            (
                "2024-01-01 24:00:00.0000005000001",
                DatetimeParseError::FieldOverflow,
            ),
            ("2024-01-01 23:59:60.1", DatetimeParseError::FieldOverflow),
            (
                "2024-01-01 23:59:60.999999",
                DatetimeParseError::FieldOverflow,
            ),
            (
                "2024-01-01 23:59:60.9999994",
                DatetimeParseError::FieldOverflow,
            ),
            (
                "2024-01-01 23:59:60.9999995",
                DatetimeParseError::FieldOverflow,
            ),
            ("294276-12-31 23:59:60", DatetimeParseError::FieldOverflow),
            ("4714-11-23 23:59:60 BC", DatetimeParseError::FieldOverflow),
            ("2024-01-01T", DatetimeParseError::MalformedFormat),
            ("2024-01-01 T", DatetimeParseError::MalformedFormat),
        ] {
            assert_eq!(parse_timestamp_detailed(text), Err(expected), "{text}");
        }
        assert_eq!(
            parse_timestamp_detailed("4714-11-24 00:00:60 BC"),
            parse_timestamp_detailed("4714-11-24 00:01:00 BC")
        );
        assert_eq!(
            parse_timestamp_detailed("294276-12-31 23:59:59.9999994"),
            Ok(PG_TIMESTAMP_END_MICROS_EXCLUSIVE - 1),
            "the final finite timestamp stays in range before the carry threshold"
        );
    }
}
