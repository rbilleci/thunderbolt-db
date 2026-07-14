//! Fixed-point decimal value and arithmetic.

/// A fixed-point decimal: an `i128` unscaled `mantissa` and a `u8` `scale`
/// (number of fractional digits). `12345.67` at scale 2 is `{ mantissa: 1234567, scale: 2 }`.
///
/// Chosen over a string representation for NUMERIC storage: 16 bytes inline (no
/// per-value heap allocation), hardware-speed compare/arithmetic on the i128
/// mantissa, a fixed width amenable to GPU residency, and — because a value is
/// reduced to a single `(mantissa, scale)` pair — canonical-by-construction
/// equality at a *given* scale. Cross-scale equality is handled by [`Decimal128::cmp`]
/// (scale-aligned), and the engine's value-index keys on a canonical scale so that
/// `1.0` and `1.00` collide. Overflow beyond the i128 mantissa range is a clean
/// errored edge (no bignum fallback) — values needing >38 significant digits are a
/// documented future milestone.
#[derive(Debug, Clone, Copy)]
pub struct Decimal128 {
    pub mantissa: i128,
    pub scale: u8,
}

/// Error raised when a `numeric` value or operation exceeds the i128 mantissa range
/// (or its column `precision`). Mirrors PostgreSQL's `22003 numeric_value_out_of_range`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("numeric field overflow")]
pub struct NumericOverflow;

impl Decimal128 {
    pub const ZERO: Self = Self {
        mantissa: 0,
        scale: 0,
    };

    pub const fn new(mantissa: i128, scale: u8) -> Self {
        Self { mantissa, scale }
    }

    /// 10^exp as i128, or `None` on overflow.
    fn pow10(exp: u8) -> Option<i128> {
        let mut acc: i128 = 1;
        for _ in 0..exp {
            acc = acc.checked_mul(10)?;
        }
        Some(acc)
    }

    /// Rescale to `target_scale`, rounding half-up (away from zero on a tie), the
    /// PostgreSQL rounding mode. Returns [`NumericOverflow`] if the rescaled mantissa
    /// leaves i128 range. Used for casts, AVG, and division to a target scale.
    pub fn rescale(self, target_scale: u8) -> Result<Self, NumericOverflow> {
        if target_scale == self.scale {
            return Ok(self);
        }
        if target_scale > self.scale {
            let factor = Self::pow10(target_scale - self.scale).ok_or(NumericOverflow)?;
            let mantissa = self.mantissa.checked_mul(factor).ok_or(NumericOverflow)?;
            return Ok(Self {
                mantissa,
                scale: target_scale,
            });
        }
        // Reducing scale: divide by 10^(drop), rounding half-up on the discarded digits.
        let drop = self.scale - target_scale;
        let factor = Self::pow10(drop).ok_or(NumericOverflow)?;
        let negative = self.mantissa < 0;
        let abs = self.mantissa.unsigned_abs();
        let factor_abs = factor as u128;
        let quotient = abs / factor_abs;
        let remainder = abs % factor_abs;
        let rounded = if remainder * 2 >= factor_abs {
            quotient + 1
        } else {
            quotient
        };
        let mantissa = i128::try_from(rounded).map_err(|_| NumericOverflow)?;
        let mantissa = if negative { -mantissa } else { mantissa };
        Ok(Self {
            mantissa,
            scale: target_scale,
        })
    }

    /// The truncated-toward-zero integer part as an `i128` — always representable, since
    /// `|mantissa / 10^scale| ≤ |mantissa|`. For `scale ≥ 39`, `10^scale` exceeds i128
    /// range while `|mantissa| < 10^39 ≤ 10^scale`, so the integer part is `0`.
    fn integer_part(&self) -> i128 {
        match Self::pow10(self.scale) {
            Some(factor) => self.mantissa / factor,
            None => 0,
        }
    }

    /// Scale-aligned ordering: compares the two values at the wider of the two scales (so
    /// `1.0` and `1.00` compare equal). This is the shared kernel behind the
    /// [`Ord`]/[`PartialOrd`]/[`PartialEq`] impls, so it MUST be a consistent total order —
    /// a broken one silently corrupts any `BTreeMap` keyed on `SqlValue` (e.g. GROUP BY).
    ///
    /// When up-aligning to the wider scale overflows i128, it resolves WITHOUT overflow:
    /// by sign, then by truncated integer part (both per-value keys, hence transitive),
    /// then by the values rounded DOWN to the narrower scale (down-rescaling only shrinks
    /// the magnitude, so it cannot overflow when the scale gap ≤ 38). This branch is only
    /// reachable comparing extreme (>~38-digit) values at *differing* scales — never for a
    /// stored column value (scale ≤ precision ≤ 38), and never inside a GROUP BY BTreeMap
    /// (whose keys share one column's scale and so always take the same-scale fast path).
    pub fn compare(&self, other: &Self) -> core::cmp::Ordering {
        if self.scale == other.scale {
            return self.mantissa.cmp(&other.mantissa);
        }
        let target = self.scale.max(other.scale);
        if let (Ok(left), Ok(right)) = (self.rescale(target), other.rescale(target)) {
            return left.mantissa.cmp(&right.mantissa);
        }
        // Exact up-alignment overflowed i128 (pathological extreme magnitudes).
        let by_sign = self.mantissa.signum().cmp(&other.mantissa.signum());
        if by_sign != core::cmp::Ordering::Equal {
            return by_sign;
        }
        let by_integer = self.integer_part().cmp(&other.integer_part());
        if by_integer != core::cmp::Ordering::Equal {
            return by_integer;
        }
        // Equal sign and integer part: discriminate the fraction at the narrower scale
        // (rescaling DOWN cannot overflow for a scale gap ≤ 38). A gap > 38 with equal
        // integer parts is unrepresentable for an i128 mantissa, so `Equal` is unreachable.
        let narrower = self.scale.min(other.scale);
        match (self.rescale(narrower), other.rescale(narrower)) {
            (Ok(left), Ok(right)) => left.mantissa.cmp(&right.mantissa),
            _ => core::cmp::Ordering::Equal,
        }
    }

    /// Whether two values are numerically equal regardless of scale (`1.0 == 1.00`).
    pub fn numeric_eq(&self, other: &Self) -> bool {
        self.compare(other) == core::cmp::Ordering::Equal
    }

    /// The canonical form: trailing decimal zeros stripped (so `1.00` → `1`, `1.050` → `1.05`).
    /// Two values that are numerically equal share one canonical `(mantissa, scale)`, so an
    /// equality value-index keyed on the canonical form collides `1.0` with `1.00`.
    pub fn canonical(self) -> Self {
        let mut mantissa = self.mantissa;
        let mut scale = self.scale;
        while scale > 0 && mantissa % 10 == 0 {
            mantissa /= 10;
            scale -= 1;
        }
        Self { mantissa, scale }
    }

    fn add_sub(self, other: Self, subtract: bool) -> Result<Self, NumericOverflow> {
        let target = self.scale.max(other.scale);
        let left = self.rescale(target)?;
        let right = other.rescale(target)?;
        let mantissa = if subtract {
            left.mantissa.checked_sub(right.mantissa)
        } else {
            left.mantissa.checked_add(right.mantissa)
        }
        .ok_or(NumericOverflow)?;
        Ok(Self {
            mantissa,
            scale: target,
        })
    }

    /// Scale-aligned addition (result scale = max of the two), overflow-checked.
    pub fn checked_add(self, other: Self) -> Result<Self, NumericOverflow> {
        self.add_sub(other, false)
    }

    /// Scale-aligned subtraction (result scale = max of the two), overflow-checked.
    pub fn checked_sub(self, other: Self) -> Result<Self, NumericOverflow> {
        self.add_sub(other, true)
    }

    /// Parse a decimal literal (`"-12.50"`, `"42"`, `"0.001"`) at `target_scale`,
    /// rounding half-up. Returns `None` on a malformed literal and on overflow.
    pub fn parse_at_scale(input: &str, target_scale: u8) -> Option<Self> {
        let parsed = Self::parse(input)?;
        parsed.rescale(target_scale).ok()
    }

    /// Parse a decimal literal, inferring `scale` from the number of fractional
    /// digits present (`"4.50"` → scale 2, `"42"` → scale 0). Returns `None` on a
    /// malformed literal or i128 overflow. This is the literal's *natural* scale; the
    /// caller rescales to a column's typmod via [`Decimal128::rescale`].
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }
        let (negative, body) = match trimmed.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
        };
        if body.is_empty() {
            return None;
        }
        let (whole, frac) = body.split_once('.').unwrap_or((body, ""));
        // A bare "." or "-." is malformed; at least one digit must be present.
        if whole.is_empty() && frac.is_empty() {
            return None;
        }
        if !whole.bytes().all(|b| b.is_ascii_digit()) || !frac.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let scale = u8::try_from(frac.len()).ok()?;
        let digits = format!("{whole}{frac}");
        // An all-empty / dot-only digit string (e.g. "." parsed to whole="" frac="")
        // is rejected above; "0" / "00" parse to magnitude 0. Parse the magnitude as
        // u128 then apply the sign, so the `i128::MIN` magnitude (2^127, one past
        // i128::MAX) round-trips with `to_decimal_string` instead of failing the parse.
        let magnitude: u128 = if digits.is_empty() {
            0
        } else {
            digits.parse::<u128>().ok()?
        };
        let mantissa = if negative {
            match i128::try_from(magnitude) {
                Ok(value) => -value,
                // 2^127 is exactly |i128::MIN|; anything larger is out of range.
                Err(_) if magnitude == (i128::MAX as u128) + 1 => i128::MIN,
                Err(_) => return None,
            }
        } else {
            i128::try_from(magnitude).ok()?
        };
        Some(Self { mantissa, scale })
    }

    /// Format to a decimal string with exactly `scale` fractional digits and a leading
    /// `-` for negatives (`{ mantissa: 1234567, scale: 2 }` → `"12345.67"`). Round-trips
    /// with [`Decimal128::parse`] at the same scale, so AVG's scale-16 output is byte-stable.
    pub fn to_decimal_string(&self) -> String {
        if self.scale == 0 {
            return self.mantissa.to_string();
        }
        let negative = self.mantissa < 0;
        let digits = self.mantissa.unsigned_abs().to_string();
        let scale = self.scale as usize;
        let (whole, frac) = if digits.len() > scale {
            let split = digits.len() - scale;
            (digits[..split].to_string(), digits[split..].to_string())
        } else {
            let mut frac = "0".repeat(scale - digits.len());
            frac.push_str(&digits);
            ("0".to_string(), frac)
        };
        format!("{}{whole}.{frac}", if negative { "-" } else { "" })
    }
}

impl core::fmt::Display for Decimal128 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_decimal_string())
    }
}

impl PartialEq for Decimal128 {
    fn eq(&self, other: &Self) -> bool {
        self.numeric_eq(other)
    }
}

impl Eq for Decimal128 {}

impl PartialOrd for Decimal128 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Decimal128 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.compare(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_decimal_inferring_natural_scale() {
        assert_eq!(
            Decimal128::parse("12345.67"),
            Some(Decimal128::new(1234567, 2))
        );
        assert_eq!(Decimal128::parse("42"), Some(Decimal128::new(42, 0)));
        assert_eq!(Decimal128::parse("0.001"), Some(Decimal128::new(1, 3)));
        assert_eq!(Decimal128::parse("-12.50"), Some(Decimal128::new(-1250, 2)));
        assert_eq!(Decimal128::parse("+7"), Some(Decimal128::new(7, 0)));
        assert_eq!(Decimal128::parse("0"), Some(Decimal128::new(0, 0)));
        assert_eq!(Decimal128::parse("0.00"), Some(Decimal128::new(0, 2)));
    }

    #[test]
    fn rejects_malformed_decimal_literals() {
        assert_eq!(Decimal128::parse(""), None);
        assert_eq!(Decimal128::parse("."), None);
        assert_eq!(Decimal128::parse("-"), None);
        assert_eq!(Decimal128::parse("1.2.3"), None);
        assert_eq!(Decimal128::parse("1e5"), None);
        assert_eq!(Decimal128::parse("abc"), None);
        assert_eq!(Decimal128::parse("12 34"), None);
    }

    #[test]
    fn formats_round_trips_with_parse_preserving_scale() {
        for literal in ["12345.67", "-12.50", "0.001", "42", "1000000.000", "-0.99"] {
            let parsed = Decimal128::parse(literal).unwrap();
            assert_eq!(parsed.to_decimal_string(), literal, "round-trip {literal}");
        }
        // Fractional magnitude smaller than scale pads with leading zeros.
        assert_eq!(Decimal128::new(5, 3).to_decimal_string(), "0.005");
        assert_eq!(Decimal128::new(-5, 3).to_decimal_string(), "-0.005");
    }

    #[test]
    fn compares_scale_aligned_so_one_point_zero_equals_one_point_zero_zero() {
        let a = Decimal128::new(10, 1); // 1.0
        let b = Decimal128::new(100, 2); // 1.00
        assert_eq!(a.cmp(&b), core::cmp::Ordering::Equal);
        assert!(a.numeric_eq(&b));
        assert_eq!(a, b); // PartialEq is scale-aligned.

        let c = Decimal128::new(101, 2); // 1.01
        assert!(a < c);
        assert!(c > b);

        let neg = Decimal128::new(-1, 0);
        let pos = Decimal128::new(1, 2);
        assert!(neg < pos);
    }

    #[test]
    fn compare_is_a_consistent_total_order_when_scale_alignment_overflows() {
        // Up-aligning to the wider scale overflows i128 (3e37 * 10^18), the case the old
        // raw-mantissa fallback got wrong. Truth: 3e37 (scale 0) > 1.6e20 (scale 18).
        let big = Decimal128::new(30_000_000_000_000_000_000_000_000_000_000_000_000, 0);
        let small = Decimal128::new(160_000_000_000_000_000_000_000_000_000_000_000_000, 18);
        assert_eq!(big.compare(&small), core::cmp::Ordering::Greater);
        assert_eq!(small.compare(&big), core::cmp::Ordering::Less); // antisymmetric
        assert_ne!(big, small);
        // Same magnitude, opposite signs still resolve by sign even at the ceiling.
        let neg_big = Decimal128::new(-30_000_000_000_000_000_000_000_000_000_000_000_000, 0);
        assert_eq!(neg_big.compare(&small), core::cmp::Ordering::Less);
        assert_eq!(small.compare(&neg_big), core::cmp::Ordering::Greater);
    }

    #[test]
    fn parse_and_format_round_trip_at_the_i128_min_boundary() {
        // i128::MIN's magnitude is 2^127 = |i128::MAX| + 1; parsing must not overflow.
        let formatted = Decimal128::new(i128::MIN, 0).to_decimal_string();
        assert_eq!(
            Decimal128::parse(&formatted),
            Some(Decimal128::new(i128::MIN, 0))
        );
        assert_eq!(
            Decimal128::new(i128::MAX, 0)
                .to_decimal_string()
                .parse::<i128>(),
            Ok(i128::MAX)
        );
        // One past i128::MIN's magnitude is still rejected.
        assert_eq!(
            Decimal128::parse("-170141183460469231731687303715884105729"),
            None
        );
    }

    #[test]
    fn rescale_rounds_half_up_postgres_style() {
        // Widening scale is exact.
        assert_eq!(
            Decimal128::new(125, 1).rescale(3).unwrap(),
            Decimal128::new(12500, 3)
        );
        // 1.25 -> scale 1 rounds half-up to 1.3.
        assert_eq!(
            Decimal128::new(125, 2).rescale(1).unwrap(),
            Decimal128::new(13, 1)
        );
        // 1.24 -> scale 1 rounds down to 1.2.
        assert_eq!(
            Decimal128::new(124, 2).rescale(1).unwrap(),
            Decimal128::new(12, 1)
        );
        // Negative rounds away from zero on a tie: -1.25 -> -1.3.
        assert_eq!(
            Decimal128::new(-125, 2).rescale(1).unwrap(),
            Decimal128::new(-13, 1)
        );
        // Half-up at the boundary: 0.5 -> scale 0 is 1.
        assert_eq!(
            Decimal128::new(5, 1).rescale(0).unwrap(),
            Decimal128::new(1, 0)
        );
    }

    #[test]
    fn parse_at_scale_rounds_to_target() {
        assert_eq!(
            Decimal128::parse_at_scale("1.005", 2),
            Some(Decimal128::new(101, 2))
        );
        assert_eq!(
            Decimal128::parse_at_scale("1.004", 2),
            Some(Decimal128::new(100, 2))
        );
        assert_eq!(
            Decimal128::parse_at_scale("5", 2),
            Some(Decimal128::new(500, 2))
        );
    }

    #[test]
    fn add_sub_align_scales_and_check_overflow() {
        // 10.50 + 0.005 = 10.505 (result scale = max scale).
        assert_eq!(
            Decimal128::new(1050, 2)
                .checked_add(Decimal128::new(5, 3))
                .unwrap(),
            Decimal128::new(10505, 3)
        );
        // 10.00 - 2.50 = 7.50.
        assert_eq!(
            Decimal128::new(1000, 2)
                .checked_sub(Decimal128::new(250, 2))
                .unwrap(),
            Decimal128::new(750, 2)
        );
        // Overflow near the i128 ceiling.
        assert_eq!(
            Decimal128::new(i128::MAX, 0).checked_add(Decimal128::new(1, 0)),
            Err(NumericOverflow)
        );
        // Rescaling overflow: widening i128::MAX by a digit overflows.
        assert_eq!(
            Decimal128::new(i128::MAX, 0).rescale(1),
            Err(NumericOverflow)
        );
    }
}
