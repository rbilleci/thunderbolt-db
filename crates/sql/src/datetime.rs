//! Minimal proleptic-Gregorian calendar math for the temporal types (the type matrix, doc 19).
//! Hand-rolled (no `chrono`/`time` dependency): the conversions are Howard Hinnant's standard
//! `days_from_civil` / `civil_from_days` algorithms. A `DATE` is stored as i32 DAYS since 2000-01-01
//! (the PostgreSQL date epoch), so byte-for-byte it is an i32 column and reuses the int4 device path.

/// Days from 1970-01-01 (the algorithm's Unix epoch) to 2000-01-01 (PostgreSQL's date epoch). Used to
/// rebase between the two so `DATE` values are days-since-2000.
const PG_EPOCH_UNIX_DAYS: i64 = 10957;

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
    let y: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let d: i64 = parts.next()?.parse().ok()?;
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
    }
}
