//! Pure shard zone-map pruning and cross-shard point-lookup shape analysis.

use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};
use crate::relational_model::{RelationalTable, ResidentDeviceInt4ColumnStats};
use crate::resident_route::BoundRelationalSelect;
use gpu_db_sql::{SelectFilterOp, SqlType, SqlValue};

/// S-d3 zone-map pruning: collect the int4 column equalities (`col = needle`) that EVERY matching row
/// must satisfy — i.e. those reachable through top-level AND nodes only. A constraint under an OR (or any
/// other node) is NOT mandatory and is skipped, so the collected set is always sound to prune on: a shard
/// whose zone map for `col` excludes `needle` cannot hold a single matching row. Conservative by
/// construction — an unrecognized shape simply yields no constraint (no pruning), never a wrong prune.
pub(super) fn mandatory_int4_equalities(expr: &ResidentExpr, out: &mut Vec<(usize, i32)>) {
    match expr {
        ResidentExpr::Binary {
            op: ResidentBinaryOp::And,
            lhs,
            rhs,
        } => {
            mandatory_int4_equalities(lhs, out);
            mandatory_int4_equalities(rhs, out);
        }
        ResidentExpr::Binary {
            op: ResidentBinaryOp::Eq,
            lhs,
            rhs,
        } => match (lhs.as_ref(), rhs.as_ref()) {
            (ResidentExpr::Column(c), ResidentExpr::Int4Literal(n))
            | (ResidentExpr::Int4Literal(n), ResidentExpr::Column(c)) => out.push((*c, *n)),
            _ => {}
        },
        _ => {}
    }
}

/// Sub-slice 3b: detect the CROSS-SHARD PK-index route's precondition — a lone top-level int4 equality
/// `col = needle` — from a (still-populated) bound select, returning `Some((filter_idx, needle))` or `None`.
/// Mirrors the retained-read (lpb) route's shape gate (`build_relational_retained_read_job`): EXACTLY one
/// equality group of one predicate, operator `Eq`, an int4 filter column, and an int4 needle. Any other
/// shape (multiple predicates, a range, a non-int4 column/value, no filter) returns `None` so the caller
/// runs the scan path unchanged. Must be called BEFORE the caller clears `bound`'s filters.
pub(crate) fn shard_point_lookup_int4_eq(
    bound: &BoundRelationalSelect,
    table: &RelationalTable,
) -> Option<(usize, i32)> {
    let filter_groups = if !bound.filter_groups.is_empty() {
        bound.filter_groups.clone()
    } else if !bound.filters.is_empty() {
        vec![bound.filters.clone()]
    } else if let Some(filter) = bound.filter.clone() {
        vec![vec![filter]]
    } else {
        return None;
    };
    if filter_groups.len() != 1 || filter_groups[0].len() != 1 {
        return None;
    }
    let (filter_idx, op, value) = filter_groups[0][0].clone();
    if op != SelectFilterOp::Eq {
        return None;
    }
    let SqlValue::Int4(needle) = value else {
        return None;
    };
    if table.columns.get(filter_idx).map(|c| c.ty) != Some(SqlType::Int4) {
        return None;
    }
    Some((filter_idx, needle))
}

/// S-d3 zone-map pruning core: does `shard_stats` (a shard's per-int4-column zone map) provably EXCLUDE
/// `needle` for the FULL-CATALOG column index `col` into `column_names`? Returns `false` (⇒ keep the
/// shard) unless the column resolves to a zone-map stat BY NAME whose `[min, max]` rules the needle out.
///
/// SOUNDNESS: `col` is a full-catalog index (from `ResidentExpr::Column`), but `shard_stats` is the
/// int4-ordinal-COMPACTED list (only int4/date/int2 columns). Resolving `column_names[col]` to a name and
/// matching the stat by that name performs the catalog→int4-ordinal translation implicitly — exactly what
/// `resident_device_int4_column_offset` does. Indexing `shard_stats[col]` directly would read the WRONG
/// column on a mixed-type table (a non-int4 column before the filter column) and could wrongly prune a
/// shard that holds matching rows. An unresolved column (out of range, or non-int4 ⇒ no stat) ⇒ `false`.
pub(super) fn shard_zone_map_excludes(
    column_names: &[&str],
    shard_stats: &[ResidentDeviceInt4ColumnStats],
    col: usize,
    needle: i32,
) -> bool {
    column_names.get(col).is_some_and(|name| {
        shard_stats
            .iter()
            .find(|s| s.name == *name)
            .is_some_and(|s| needle < s.min || needle > s.max)
    })
}
#[cfg(test)]
mod zone_map_prune_tests {
    use super::{
        mandatory_int4_equalities, shard_zone_map_excludes, ResidentBinaryOp, ResidentExpr,
    };
    use crate::relational_model::ResidentDeviceInt4ColumnStats;

    fn stat(name: &str, min: i32, max: i32) -> ResidentDeviceInt4ColumnStats {
        ResidentDeviceInt4ColumnStats {
            name: name.to_string(),
            min,
            max,
        }
    }

    /// A load-bearing regression guard: on a MIXED-type table the filter
    /// column's FULL-CATALOG index differs from its int4 ordinal, so the prune MUST resolve the zone-map
    /// stat by NAME, not by indexing the int4-ordinal-compacted stat list with the catalog index.
    ///
    /// Layout: `flag BOOL(0), a INT(1), b INT(2), c INT(3)` -> catalog names `[flag,a,b,c]`, but the int4
    /// stat list is only `[a,b,c]`. Construct RANGES so the correct column (`a`) and the wrong column that
    /// a naive `stats[col]`/compacted-index lookup would hit (`b`, at compacted ordinal 1 == catalog idx 1)
    /// DISAGREE on the needle: `a` INCLUDES it, `b` EXCLUDES it. The by-name resolver must NOT exclude.
    #[test]
    fn mixed_type_prune_resolves_column_by_name_not_ordinal() {
        let column_names = ["flag", "a", "b", "c"];
        // a=[128,191] (includes 137), b=[1128,1191] (EXCLUDES 137), c=[128,191].
        let stats = [
            stat("a", 128, 191),
            stat("b", 1128, 1191),
            stat("c", 128, 191),
        ];

        // Correct: `WHERE a = 137` -> catalog col 1 -> name "a" -> [128,191] INCLUDES 137 -> NOT excluded.
        assert!(
            !shard_zone_map_excludes(&column_names, &stats, 1, 137),
            "a=137 must NOT prune this shard (a's range includes 137)"
        );
        // The bug it guards against: indexing the compacted stat list by catalog idx 1 hits "b" [1128,1191],
        // which WOULD exclude 137 -> a wrongly dropped shard. Prove that wrong column really would exclude:
        assert!(
            137 < stats[1].min,
            "sanity: the mis-resolved column b WOULD have excluded 137 (that is the bug)"
        );

        // Correct positive prune: `WHERE b = 137` -> catalog col 2 -> name "b" [1128,1191] -> excluded.
        assert!(
            shard_zone_map_excludes(&column_names, &stats, 2, 137),
            "b=137 must prune (b's range excludes 137)"
        );
        // `WHERE b = 1150` -> name "b" includes 1150 -> not excluded.
        assert!(!shard_zone_map_excludes(&column_names, &stats, 2, 1150));
        // `WHERE c = 137` -> catalog col 3 -> name "c" [128,191] includes -> not excluded (a raw stats[3]
        // would even panic/out-of-range; by-name it resolves correctly).
        assert!(!shard_zone_map_excludes(&column_names, &stats, 3, 137));

        // Unresolvable / non-int4 columns are always KEPT (never a wrong prune):
        //  - the BOOL column `flag` (catalog 0) has no int4 stat -> not excluded.
        assert!(!shard_zone_map_excludes(&column_names, &stats, 0, 137));
        //  - an out-of-range catalog index -> not excluded.
        assert!(!shard_zone_map_excludes(&column_names, &stats, 99, 137));
        //  - an empty zone map (e.g. a benchmark shard) -> never pruned.
        assert!(!shard_zone_map_excludes(&column_names, &[], 1, 137));
    }

    /// `mandatory_int4_equalities` collects only equalities that EVERY matching row must satisfy: those
    /// reachable through top-level AND nodes. An OR (or any non-equality) contributes nothing, so the
    /// collected set is always sound to prune on.
    #[test]
    fn mandatory_equalities_only_under_top_level_and() {
        let col = |i| Box::new(ResidentExpr::Column(i));
        let lit = |n| Box::new(ResidentExpr::Int4Literal(n));
        let eq = |i, n| ResidentExpr::Binary {
            op: ResidentBinaryOp::Eq,
            lhs: col(i),
            rhs: lit(n),
        };
        let bin = |op, l, r| ResidentExpr::Binary {
            op,
            lhs: Box::new(l),
            rhs: Box::new(r),
        };

        // `a = 1` -> one constraint.
        let mut out = Vec::new();
        mandatory_int4_equalities(&eq(1, 1), &mut out);
        assert_eq!(out, vec![(1, 1)]);

        // `a = 1 AND b = 2` -> both constraints (a match must satisfy both).
        let mut out = Vec::new();
        mandatory_int4_equalities(&bin(ResidentBinaryOp::And, eq(1, 1), eq(2, 2)), &mut out);
        assert_eq!(out, vec![(1, 1), (2, 2)]);

        // `a = 1 OR b = 2` -> NO mandatory constraint (a match may satisfy only one).
        let mut out = Vec::new();
        mandatory_int4_equalities(&bin(ResidentBinaryOp::Or, eq(1, 1), eq(2, 2)), &mut out);
        assert!(
            out.is_empty(),
            "a top-level OR yields no mandatory equality (no prune)"
        );

        // `a = 1 AND (b = 2 OR c = 3)` -> only `a = 1` is mandatory.
        let mut out = Vec::new();
        let or = bin(ResidentBinaryOp::Or, eq(2, 2), eq(3, 3));
        mandatory_int4_equalities(&bin(ResidentBinaryOp::And, eq(1, 1), or), &mut out);
        assert_eq!(out, vec![(1, 1)], "only the AND-side equality is mandatory");

        // A literal-on-the-left equality (`1 = a`) is still recognized.
        let mut out = Vec::new();
        mandatory_int4_equalities(
            &ResidentExpr::Binary {
                op: ResidentBinaryOp::Eq,
                lhs: lit(7),
                rhs: col(4),
            },
            &mut out,
        );
        assert_eq!(out, vec![(4, 7)]);
    }
}
