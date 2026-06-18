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

/// An ASCII-digits-only unsigned component (a date/time field), as i64. Rejects an empty string and a
/// leading sign / whitespace that Rust's integer parse would otherwise accept (PG rejects `+2024`).
fn ascii_digits(s: &str) -> Option<i64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Days from a proleptic-Gregorian civil date to 1970-01-01 (Howard Hinnant's `days_from_civil`).
/// `m` is `[1, 12]`, `d` is `[1, 31]`; the result is exact for any in-range year.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // Mar=0 .. Feb=11
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
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

/// Parse a PostgreSQL `DATE` literal in ISO `YYYY-MM-DD` form to i32 days since 2000-01-01. Returns
/// `None` for a malformed, out-of-calendar (e.g. `2024-02-30`), or i32-out-of-range date; the caller
/// turns that into PG's "invalid input syntax for type date". Leading/trailing ASCII whitespace is
/// tolerated. Non-ISO styles and BC/negative years are not accepted yet (a follow-on).
pub fn parse_date(text: &str) -> Option<i32> {
    let trimmed = text.trim();
    let mut parts = trimmed.splitn(3, '-');
    let y = ascii_digits(parts.next()?)?;
    let m = ascii_digits(parts.next()?)?;
    let d = ascii_digits(parts.next()?)?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let days = days_from_civil(y, m, d) - PG_EPOCH_UNIX_DAYS;
    // Round-trip to reject a day that does not exist in that month (e.g. 2024-02-30 -> 2024-03-01).
    if civil_from_days(days + PG_EPOCH_UNIX_DAYS) != (y, m, d) {
        return None;
    }
    i32::try_from(days).ok()
}

/// Format i32 days since 2000-01-01 as the ISO `YYYY-MM-DD` text PostgreSQL emits.
pub fn format_date(days: i32) -> String {
    let (y, m, d) = civil_from_days(i64::from(days) + PG_EPOCH_UNIX_DAYS);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Parse a time-of-day `HH:MM[:SS[.ffffff]]` to microseconds within the day. Seconds default to 0
/// (`HH:MM`). The fractional part is right-padded / truncated to 6 digits (microseconds).
fn parse_time_of_day(text: &str) -> Option<i64> {
    let (hms, frac) = match text.split_once('.') {
        Some((hms, frac)) => (hms, Some(frac)),
        None => (text, None),
    };
    let mut parts = hms.split(':');
    let h = ascii_digits(parts.next()?)?;
    let m = ascii_digits(parts.next()?)?;
    let s = match parts.next() {
        Some(sec) => ascii_digits(sec)?,
        None => 0,
    };
    if parts.next().is_some() || h > 23 || m > 59 || s > 59 {
        return None;
    }
    let frac_micros = match frac {
        None => 0,
        Some(f) => {
            if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            // Right-pad (or truncate) the fractional digits to exactly 6 -> microseconds.
            let mut micros = [b'0'; 6];
            for (slot, byte) in micros.iter_mut().zip(f.bytes()) {
                *slot = byte;
            }
            std::str::from_utf8(&micros).ok()?.parse::<i64>().ok()?
        }
    };
    Some(h * 3_600_000_000 + m * 60_000_000 + s * MICROS_PER_SEC + frac_micros)
}

/// Parse a PostgreSQL `TIMESTAMP` literal `YYYY-MM-DD[ T]HH:MM[:SS[.ffffff]]` (the time part optional,
/// defaulting to midnight) to i64 MICROSECONDS since 2000-01-01 00:00:00. `None` for a malformed,
/// out-of-calendar, or i64-out-of-range timestamp; the caller raises PG's "invalid input syntax for
/// type timestamp". Non-ISO styles and time zones are not accepted yet (a follow-on).
pub fn parse_timestamp(text: &str) -> Option<i64> {
    let trimmed = text.trim();
    let (date_part, time_part) = match trimmed.find([' ', 'T']) {
        Some(pos) => (&trimmed[..pos], Some(trimmed[pos + 1..].trim())),
        None => (trimmed, None),
    };
    let days = i64::from(parse_date(date_part)?);
    let tod = match time_part {
        Some(t) if !t.is_empty() => parse_time_of_day(t)?,
        _ => 0,
    };
    days.checked_mul(MICROS_PER_DAY)?.checked_add(tod)
}

/// Format i64 microseconds since 2000-01-01 00:00:00 as the text PostgreSQL emits: `YYYY-MM-DD
/// HH:MM:SS`, plus a `.ffffff` fractional part (trailing zeros trimmed) when non-zero.
pub fn format_timestamp(micros: i64) -> String {
    let days = micros.div_euclid(MICROS_PER_DAY);
    let tod = micros.rem_euclid(MICROS_PER_DAY);
    let (y, mo, d) = civil_from_days(days + PG_EPOCH_UNIX_DAYS);
    let secs = tod / MICROS_PER_SEC;
    let frac = tod % MICROS_PER_SEC;
    let (h, mi, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    let mut out = format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}");
    if frac != 0 {
        out.push('.');
        out.push_str(format!("{frac:06}").trim_end_matches('0'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_epoch_and_neighbors() {
        assert_eq!(parse_date("2000-01-01"), Some(0), "the PG date epoch is day 0");
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
        assert_eq!(parse_date("2000-02-29"), parse_date("2000-02-29"), "2000 IS a leap year");
        assert!(parse_date("2000-02-29").is_some(), "2000 is divisible by 400 -> leap");
        assert_eq!(parse_date("1900-02-29"), None, "1900 is divisible by 100 not 400 -> not leap");
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
        assert_eq!(parse_date("2024- 01-15"), None, "embedded space in a component");
    }

    #[test]
    fn timestamp_epoch_and_units() {
        assert_eq!(parse_timestamp("2000-01-01 00:00:00"), Some(0), "the timestamp epoch is micro 0");
        assert_eq!(parse_timestamp("2000-01-01 00:00:01"), Some(1_000_000), "one second");
        assert_eq!(parse_timestamp("2000-01-01 00:01:00"), Some(60_000_000), "one minute");
        assert_eq!(parse_timestamp("2000-01-01 01:00:00"), Some(3_600_000_000), "one hour");
        assert_eq!(parse_timestamp("2000-01-02 00:00:00"), Some(MICROS_PER_DAY), "one day");
        // no time part -> midnight; 'T' separator accepted; HH:MM defaults seconds to 0.
        assert_eq!(parse_timestamp("2000-01-02"), Some(MICROS_PER_DAY), "date-only => midnight");
        assert_eq!(
            parse_timestamp("2000-01-01T00:01:00"),
            Some(60_000_000),
            "'T' separator"
        );
        assert_eq!(parse_timestamp("2000-01-01 00:01"), Some(60_000_000), "HH:MM (no seconds)");
    }

    #[test]
    fn timestamp_fractional_seconds() {
        assert_eq!(parse_timestamp("2000-01-01 00:00:00.5"), Some(500_000), ".5 => 500000 us");
        assert_eq!(parse_timestamp("2000-01-01 00:00:00.000001"), Some(1), "one microsecond");
        assert_eq!(parse_timestamp("2000-01-01 00:00:00.123456"), Some(123_456));
        // PG trims trailing zeros from the fractional part on output, omits it when zero.
        assert_eq!(format_timestamp(0), "2000-01-01 00:00:00");
        assert_eq!(format_timestamp(500_000), "2000-01-01 00:00:00.5");
        assert_eq!(format_timestamp(123_456), "2000-01-01 00:00:00.123456");
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
        assert_eq!(parse_timestamp("2024-01-15 10:00:60"), None, "second 60");
        assert_eq!(parse_timestamp("2024-13-01 10:00:00"), None, "bad date part");
        assert_eq!(parse_timestamp("2024-01-15 10:00:00:00"), None, "too many time parts");
        assert_eq!(parse_timestamp("2024-01-15 10:xx:00"), None, "non-numeric minute");
        assert_eq!(parse_timestamp("2024-01-15 10:00:00.abc"), None, "non-numeric fraction");
    }
}
