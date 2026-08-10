//! Deterministic SQL diagnostic arbitration for pre-WAL INSERT constraints.
//!
//! This leaf owns only the terminal candidate representation, its total ordering, and conversion
//! to established SQL errors.  Constraint operators remain in their respective device-proof
//! modules; this module neither evaluates constraints nor participates in WAL, residency, or
//! publication.

use crate::EngineError;

/// SQL constraint class precedence after incoming-row order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ConstraintPhase {
    PrimaryKeyNull = 0,
    Check = 1,
    Duplicate = 2,
}

/// Stable order within one SQL constraint class.
#[cfg_attr(test, derive(Clone))]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ConstraintOrdinal {
    Attnum(i16),
    Check(String, usize),
    Index(usize),
}

/// The established diagnostic emitted for a chosen candidate.
#[cfg_attr(test, derive(Clone))]
enum ConstraintDiagnostic {
    NotNull { table: String, column: String },
    Check { table: String, name: String },
    Unique { name: String },
}

/// A bounded terminal from a device operation. Candidate order is exactly `(row, phase,
/// ordinal)`: CHECK ordinal is relcache lexical `(name, raw catalog ordinal)`, PRIMARY KEY NULL
/// uses attnum, and duplicate uses the raw index vector ordinal.
#[cfg_attr(test, derive(Clone))]
pub(super) struct ConstraintCandidate {
    row: u32,
    phase: ConstraintPhase,
    ordinal: ConstraintOrdinal,
    diagnostic: ConstraintDiagnostic,
}

impl ConstraintCandidate {
    pub(super) fn primary_key_null(row: u32, table: String, column: String, attnum: i16) -> Self {
        Self {
            row,
            phase: ConstraintPhase::PrimaryKeyNull,
            ordinal: ConstraintOrdinal::Attnum(attnum),
            diagnostic: ConstraintDiagnostic::NotNull { table, column },
        }
    }

    pub(super) fn check(row: u32, table: String, name: String, raw_ordinal: usize) -> Self {
        Self {
            row,
            phase: ConstraintPhase::Check,
            ordinal: ConstraintOrdinal::Check(name.clone(), raw_ordinal),
            diagnostic: ConstraintDiagnostic::Check { table, name },
        }
    }

    /// Preserve the historical iterator-minimum rule for every producer of constraint terminals.
    pub(super) fn merge(candidates: impl IntoIterator<Item = Self>) -> Option<Self> {
        candidates.into_iter().min_by(|left, right| {
            (left.row, left.phase, &left.ordinal).cmp(&(right.row, right.phase, &right.ordinal))
        })
    }

    #[allow(dead_code)] // current-generation reservation diagnostic ordering
    pub(super) fn unique(row: u32, index_ordinal: usize, name: String) -> Self {
        Self {
            row,
            phase: ConstraintPhase::Duplicate,
            ordinal: ConstraintOrdinal::Index(index_ordinal),
            diagnostic: ConstraintDiagnostic::Unique { name },
        }
    }

    #[allow(dead_code)] // current-generation reservation diagnostic ordering
    pub(super) fn choose(left: Option<Self>, right: Option<Self>) -> Option<Self> {
        match (left, right) {
            (Some(left), Some(right)) => Self::merge([left, right]),
            (left, right) => left.or(right),
        }
    }

    pub(super) fn into_error(self) -> EngineError {
        match self.diagnostic {
            ConstraintDiagnostic::NotNull { table, column } => {
                EngineError::NotNullViolation(format!(
                    "null value in column \"{column}\" of relation \"{table}\" violates not-null constraint"
                ))
            }
            ConstraintDiagnostic::Check { table, name } => EngineError::CheckViolation(format!(
                "new row for relation \"{table}\" violates check constraint \"{name}\""
            )),
            ConstraintDiagnostic::Unique { name } => EngineError::UniqueViolation(format!(
                "duplicate key value violates unique constraint \"{name}\""
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn primary_key_null(row: u32, attnum: i16, column: &str) -> ConstraintCandidate {
        ConstraintCandidate {
            row,
            phase: ConstraintPhase::PrimaryKeyNull,
            ordinal: ConstraintOrdinal::Attnum(attnum),
            diagnostic: ConstraintDiagnostic::NotNull {
                table: "arbitration".to_string(),
                column: column.to_string(),
            },
        }
    }

    fn check(row: u32, name: &str, raw_ordinal: usize) -> ConstraintCandidate {
        ConstraintCandidate {
            row,
            phase: ConstraintPhase::Check,
            ordinal: ConstraintOrdinal::Check(name.to_string(), raw_ordinal),
            diagnostic: ConstraintDiagnostic::Check {
                table: "arbitration".to_string(),
                name: name.to_string(),
            },
        }
    }

    fn duplicate(row: u32, index_ordinal: usize, name: &str) -> ConstraintCandidate {
        ConstraintCandidate {
            row,
            phase: ConstraintPhase::Duplicate,
            ordinal: ConstraintOrdinal::Index(index_ordinal),
            diagnostic: ConstraintDiagnostic::Unique {
                name: name.to_string(),
            },
        }
    }

    fn diagnostic_name(candidate: &ConstraintCandidate) -> &str {
        match &candidate.diagnostic {
            ConstraintDiagnostic::NotNull { column, .. } => column,
            ConstraintDiagnostic::Check { name, .. } | ConstraintDiagnostic::Unique { name } => {
                name
            }
        }
    }

    fn assert_pair_permutations(
        lower: ConstraintCandidate,
        higher: ConstraintCandidate,
        expected_name: &str,
    ) {
        for (left, right) in [(lower.clone(), higher.clone()), (higher, lower)] {
            let chosen = ConstraintCandidate::choose(Some(left), Some(right))
                .expect("two candidates retain one minimum");
            assert_eq!(diagnostic_name(&chosen), expected_name);
        }
    }

    fn assert_three_permutations(candidates: [ConstraintCandidate; 3], expected_name: &str) {
        for first in 0..candidates.len() {
            for second in 0..candidates.len() {
                if second == first {
                    continue;
                }
                for third in 0..candidates.len() {
                    if third == first || third == second {
                        continue;
                    }
                    let chosen = ConstraintCandidate::merge([
                        candidates[first].clone(),
                        candidates[second].clone(),
                        candidates[third].clone(),
                    ])
                    .expect("three candidates retain one minimum");
                    assert_eq!(diagnostic_name(&chosen), expected_name);
                }
            }
        }
    }

    #[test]
    fn lower_row_wins_every_cross_phase_pair_and_permutation() {
        let earlier = [
            primary_key_null(7, 9, "earlier_not_null"),
            check(7, "earlier_check", 9),
            duplicate(7, 9, "earlier_unique"),
        ];
        let later = [
            primary_key_null(8, 1, "later_not_null"),
            check(8, "later_check", 1),
            duplicate(8, 1, "later_unique"),
        ];

        for lower in &earlier {
            for higher in &later {
                assert_pair_permutations(lower.clone(), higher.clone(), diagnostic_name(lower));
            }
        }
        assert_three_permutations(
            [
                duplicate(7, 9, "earliest_row"),
                primary_key_null(8, 1, "later_not_null"),
                check(8, "later_check", 1),
            ],
            "earliest_row",
        );
    }

    #[test]
    fn phase_precedence_wins_all_same_row_pair_and_permutations() {
        let candidates = [
            primary_key_null(7, 9, "not_null"),
            check(7, "check", 9),
            duplicate(7, 9, "unique"),
        ];

        assert_pair_permutations(candidates[0].clone(), candidates[1].clone(), "not_null");
        assert_pair_permutations(candidates[0].clone(), candidates[2].clone(), "not_null");
        assert_pair_permutations(candidates[1].clone(), candidates[2].clone(), "check");
        assert_three_permutations(
            [
                candidates[0].clone(),
                candidates[2].clone(),
                candidates[1].clone(),
            ],
            "not_null",
        );
    }

    #[test]
    fn ordinal_precedence_wins_same_row_and_phase_pair_and_permutations() {
        let primary = [
            primary_key_null(7, 1, "primary_one"),
            primary_key_null(7, 2, "primary_two"),
            primary_key_null(7, 3, "primary_three"),
        ];
        let checks = [
            check(7, "alpha", 1),
            check(7, "alpha", 2),
            check(7, "bravo", 0),
        ];
        let duplicates = [
            duplicate(7, 1, "unique_one"),
            duplicate(7, 2, "unique_two"),
            duplicate(7, 3, "unique_three"),
        ];
        for candidates in [&primary, &checks, &duplicates] {
            for lower in 0..candidates.len() {
                for higher in (lower + 1)..candidates.len() {
                    assert_pair_permutations(
                        candidates[lower].clone(),
                        candidates[higher].clone(),
                        diagnostic_name(&candidates[lower]),
                    );
                }
            }
        }
        assert_three_permutations(primary, "primary_one");
        assert_three_permutations(checks, "alpha");
        assert_three_permutations(duplicates, "unique_one");
    }

    #[test]
    fn diagnostic_conversion_preserves_established_sql_errors() {
        assert!(matches!(
            primary_key_null(7, 1, "id").into_error(),
            EngineError::NotNullViolation(message)
                if message == "null value in column \"id\" of relation \"arbitration\" violates not-null constraint"
        ));
        assert!(matches!(
            check(7, "positive", 1).into_error(),
            EngineError::CheckViolation(message)
                if message == "new row for relation \"arbitration\" violates check constraint \"positive\""
        ));
        assert!(matches!(
            duplicate(7, 1, "arbitration_id_key").into_error(),
            EngineError::UniqueViolation(message)
                if message == "duplicate key value violates unique constraint \"arbitration_id_key\""
        ));
    }
}
